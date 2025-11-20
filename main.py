"""
Main SMC Trading System
Implements the prototype flow from ins.txt:
1. Read OHLCV & orderbook
2. detect_orderblocks() -> list of candidates
3. For each candidate:
     features = extract_features(candidate)
     p = scorer.predict_proba(features)
     regime = regime_model.predict(features)
     if p > threshold and regime in allowed:
         place_order_with_risk_controls()
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from data_ingestion import DataIngestion
from smc_detector import SMCDetector, OrderBlock
from feature_extraction import FeatureExtractor
from models import RegimeDetector, SignalScorer
from risk_manager import RiskManager
from execution import ExecutionEngine
import config


class SMCTradingSystem:
    """Main trading system coordinating all components"""
    
    def __init__(self, symbol: str = None, timeframe: str = None):
        self.symbol = symbol or config.SYMBOL
        self.timeframe = timeframe or config.TIMEFRAME
        
        # Initialize components
        self.data_ingestion = DataIngestion(self.symbol, self.timeframe)
        self.smc_detector = SMCDetector(
            lookback=config.ORDERBLOCK_LOOKBACK,
            min_size=config.MIN_ORDERBLOCK_SIZE
        )
        self.feature_extractor = FeatureExtractor(atr_period=config.ATR_PERIOD)
        self.regime_detector = RegimeDetector(use_ml=config.USE_ML_REGIME_DETECTOR)
        self.scorer = SignalScorer()
        self.risk_manager = RiskManager(
            max_position_size=config.MAX_POSITION_SIZE,
            stop_loss_atr_mult=config.STOP_LOSS_ATR_MULTIPLIER,
            take_profit_atr_mult=config.TAKE_PROFIT_ATR_MULTIPLIER,
            max_daily_loss=config.MAX_DAILY_LOSS,
            max_concurrent=config.MAX_CONCURRENT_POSITIONS
        )
        self.execution_engine = ExecutionEngine(
            use_limit_orders=config.USE_LIMIT_ORDERS,
            slippage_estimate=config.SLIPPAGE_ESTIMATE,
            commission_rate=config.COMMISSION_RATE
        )
        
        self.data = None
        self.capital = 100000  # Starting capital
    
    def load_data(self, filepath: Optional[str] = None):
        """Load OHLCV data"""
        self.data = self.data_ingestion.load_ohlcv(filepath)
        print(f"Loaded {len(self.data)} bars of data")
        return self.data
    
    def train_models(self):
        """Train the scoring model and regime detector on historical data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        print("Training models...")
        
        # Train regime detector first (Phase 2)
        self.regime_detector.train(self.data, lookback=200)
        
        # Detect orderblocks
        candidates = self.smc_detector.detect_orderblocks(self.data)
        print(f"Detected {len(candidates)} orderblock candidates")
        
        if len(candidates) < 10:
            print("Warning: Not enough candidates for training")
            return
        
        # Extract features
        feature_matrix = self.feature_extractor.create_feature_matrix(
            candidates, self.data
        )
        
        if len(feature_matrix) < 10:
            print("Warning: Not enough valid features for training")
            return
        
        # Generate labels
        labels = self.scorer.generate_training_labels(
            self.data, candidates, horizon=10
        )
        
        # Filter to matching lengths
        min_len = min(len(feature_matrix), len(labels))
        feature_matrix = feature_matrix.iloc[:min_len]
        labels = labels[:min_len]
        
        # Train scorer
        self.scorer.train(feature_matrix, labels)
        print(f"Trained scorer on {len(labels)} samples")
        print(f"Positive samples: {labels.sum()}, Negative samples: {len(labels) - labels.sum()}")
    
    def run_backtest(self) -> Dict:
        """
        Run backtest on historical data
        
        Returns:
            Dictionary with backtest results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        print("\nRunning backtest...")
        
        # Detect all candidates
        candidates = self.smc_detector.detect_orderblocks(self.data)
        print(f"Found {len(candidates)} orderblock candidates")
        
        trades = []
        
        # Process each candidate
        for i, candidate in enumerate(candidates):
            try:
                # Get index of candidate
                candidate_idx = self.data.index.get_loc(candidate.timestamp)
                
                # Need future bars for feature extraction and execution
                if candidate_idx + 10 >= len(self.data):
                    continue
                
                # Extract features at a future bar (simulate real-time)
                current_idx = candidate_idx + 5
                features = self.feature_extractor.extract_features(
                    candidate, self.data, current_idx
                )
                
                # Convert to DataFrame for prediction
                feature_df = pd.DataFrame([features])
                
                # Get probability from scorer
                prob = self.scorer.predict_proba(feature_df)[0]
                
                # Get regime (with ensemble if configured)
                ensemble = getattr(config, 'USE_ENSEMBLE_REGIME', False)
                regime = self.regime_detector.detect_regime(self.data, current_idx, ensemble=ensemble)
                
                # Trade decision logic
                if (prob > config.SCORER_PROBABILITY_THRESHOLD and 
                    regime in config.ALLOWED_REGIMES and
                    self.risk_manager.can_trade(self.capital)):
                    
                    # Calculate entry and stops
                    current_price = self.data.iloc[current_idx]['close']
                    atr = self.feature_extractor._calculate_atr(self.data).iloc[current_idx]
                    
                    if candidate.is_bullish:
                        entry_price = candidate.price_low
                        side = 'buy'
                    else:
                        entry_price = candidate.price_high
                        side = 'sell'
                    
                    stops = self.risk_manager.calculate_stops(
                        entry_price, atr, candidate.is_bullish
                    )
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        self.capital, entry_price, stops['stop_loss']
                    )
                    
                    # Execute trade
                    fill = self.execution_engine.place_order_with_risk_controls(
                        self.symbol, side, position_size, entry_price,
                        stops['stop_loss'], stops['take_profit'], current_price
                    )
                    
                    self.risk_manager.open_position()
                    
                    # Simulate trade outcome
                    outcome = self._simulate_trade_outcome(
                        current_idx, entry_price, stops, candidate.is_bullish
                    )
                    
                    trades.append({
                        'timestamp': candidate.timestamp,
                        'side': side,
                        'entry_price': fill.fill_price,
                        'position_size': position_size,
                        'stop_loss': stops['stop_loss'],
                        'take_profit': stops['take_profit'],
                        'probability': prob,
                        'regime': regime,
                        'outcome': outcome,
                        'pnl': outcome['pnl']
                    })
                    
                    self.capital += outcome['pnl']
                    self.risk_manager.update_pnl(outcome['pnl'])
                    self.risk_manager.close_position()
                    
            except (KeyError, IndexError) as e:
                continue
        
        # Calculate metrics
        results = self._calculate_backtest_metrics(trades)
        return results
    
    def _simulate_trade_outcome(self, entry_idx: int, entry_price: float,
                                stops: Dict, is_long: bool) -> Dict:
        """
        Simulate trade outcome over next bars
        
        Args:
            entry_idx: Index where trade was entered
            entry_price: Entry price
            stops: Dictionary with stop_loss and take_profit
            is_long: True for long, False for short
            
        Returns:
            Dictionary with outcome
        """
        # Look at next 20 bars
        max_bars = min(20, len(self.data) - entry_idx - 1)
        future_data = self.data.iloc[entry_idx+1:entry_idx+1+max_bars]
        
        for i, (timestamp, bar) in enumerate(future_data.iterrows()):
            if is_long:
                # Check stop loss
                if bar['low'] <= stops['stop_loss']:
                    pnl = stops['stop_loss'] - entry_price
                    return {'hit': 'stop_loss', 'bars': i+1, 'pnl': pnl}
                # Check take profit
                if bar['high'] >= stops['take_profit']:
                    pnl = stops['take_profit'] - entry_price
                    return {'hit': 'take_profit', 'bars': i+1, 'pnl': pnl}
            else:
                # Check stop loss
                if bar['high'] >= stops['stop_loss']:
                    pnl = entry_price - stops['stop_loss']
                    return {'hit': 'stop_loss', 'bars': i+1, 'pnl': pnl}
                # Check take profit
                if bar['low'] <= stops['take_profit']:
                    pnl = entry_price - stops['take_profit']
                    return {'hit': 'take_profit', 'bars': i+1, 'pnl': pnl}
        
        # No hit, exit at last bar
        exit_price = future_data.iloc[-1]['close']
        pnl = (exit_price - entry_price) if is_long else (entry_price - exit_price)
        return {'hit': 'timeout', 'bars': max_bars, 'pnl': pnl}
    
    def _calculate_backtest_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'sharpe_ratio': 0
            }
        
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0,
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) if len(pnls) > 1 and np.std(pnls) != 0 else 0,
            'final_capital': self.capital
        }


def main():
    """Main entry point demonstrating the system"""
    print("=" * 60)
    print("SMC Trading System - Production-Grade Prototype")
    print("=" * 60)
    
    # Initialize system
    system = SMCTradingSystem()
    
    # Load data (using generated sample data)
    system.load_data()
    
    # Train models
    system.train_models()
    
    # Run backtest
    results = system.run_backtest()
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    exec_summary = system.execution_engine.get_fills_summary()
    for metric, value in exec_summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
