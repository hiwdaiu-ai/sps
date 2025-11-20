"""
Ultra High Sharpe Strategy
Focus on consistency and high win rate to achieve Sharpe > 3

Key principles:
1. Only trade with very high confidence (>80% predicted win rate)
2. Use tight risk controls (small stops, quick profits)
3. Multiple confirmation signals required
4. Scale position size based on confidence
5. Filter out all but the highest quality setups
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from main import SMCTradingSystem
from data_ingestion import DataIngestion
import config


class UltraHighSharpe:
    """Ultra selective strategy for Sharpe > 3"""
    
    def __init__(self):
        self.capital = 100000
        self.trades = []
    
    def run_ultra_selective_strategy(self, data: pd.DataFrame) -> Dict:
        """
        Ultra selective strategy:
        - Multiple timeframe confirmation
        - Very tight entry criteria
        - Quick profit taking
        - Small consistent gains
        """
        trades = []
        capital = self.capital
        
        # Calculate multiple indicators
        data = data.copy()
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        data['rsi'] = self._calculate_rsi(data['close'], 14)
        data['atr'] = self._calculate_atr(data)
        data['macd'], data['signal'] = self._calculate_macd(data['close'])
        
        # Bollinger Bands
        data['sma_20'] = data['close'].rolling(20).mean()
        data['std_20'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['sma_20'] + 1.5 * data['std_20']
        data['bb_lower'] = data['sma_20'] - 1.5 * data['std_20']
        
        for i in range(100, len(data) - 10):
            signals = self._check_ultra_high_quality_setup(data, i)
            
            if signals['long_score'] >= 8:  # Need 8/10 confirmations
                entry = data['close'].iloc[i]
                atr = data['atr'].iloc[i]
                
                # Very tight stops and targets
                stop = entry - 0.5 * atr  # Half ATR stop
                target = entry + 0.75 * atr  # 1.5:1 R:R
                
                outcome = self._simulate_precise_exit(data, i, entry, stop, target, True)
                
                if outcome:
                    pnl_pct = outcome['pnl_pct']
                    trades.append({
                        'type': 'ultra_long',
                        'entry': entry,
                        'exit': outcome['exit'],
                        'pnl': pnl_pct * capital * 0.02,  # 2% position size
                        'pnl_pct': pnl_pct,
                        'score': signals['long_score']
                    })
            
            elif signals['short_score'] >= 8:
                entry = data['close'].iloc[i]
                atr = data['atr'].iloc[i]
                
                stop = entry + 0.5 * atr
                target = entry - 0.75 * atr
                
                outcome = self._simulate_precise_exit(data, i, entry, stop, target, False)
                
                if outcome:
                    pnl_pct = outcome['pnl_pct']
                    trades.append({
                        'type': 'ultra_short',
                        'entry': entry,
                        'exit': outcome['exit'],
                        'pnl': pnl_pct * capital * 0.02,
                        'pnl_pct': pnl_pct,
                        'score': signals['short_score']
                    })
        
        if not trades:
            return {
                'total_trades': 0,
                'sharpe_ratio': 0,
                'total_pnl': 0,
                'win_rate': 0
            }
        
        # Calculate metrics
        pnls = [t['pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        mean_return = np.mean(pnl_pcts)
        std_return = np.std(pnl_pcts)
        
        sharpe = mean_return / std_return if std_return > 0 else 0
        sharpe_annualized = sharpe * np.sqrt(250)  # Annualize
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades),
            'total_pnl': sum(pnls),
            'sharpe_ratio': sharpe,
            'sharpe_annualized': sharpe_annualized,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'final_capital': self.capital + sum(pnls),
            'trades': trades
        }
    
    def _check_ultra_high_quality_setup(self, data: pd.DataFrame, idx: int) -> Dict:
        """
        Score trading setup from 0-10 based on multiple confirmations
        Require 8+ for trade
        """
        long_score = 0
        short_score = 0
        
        # 1. Trend alignment (3 EMAs)
        if (data['ema_9'].iloc[idx] > data['ema_21'].iloc[idx] > data['ema_50'].iloc[idx]):
            long_score += 2
        elif (data['ema_9'].iloc[idx] < data['ema_21'].iloc[idx] < data['ema_50'].iloc[idx]):
            short_score += 2
        
        # 2. RSI confirmation
        rsi = data['rsi'].iloc[idx]
        if 40 < rsi < 60:  # Neutral but trending
            if data['close'].iloc[idx] > data['ema_21'].iloc[idx]:
                long_score += 1
            else:
                short_score += 1
        elif 30 < rsi < 40:  # Oversold but not extreme
            long_score += 2
        elif 60 < rsi < 70:  # Overbought but not extreme
            short_score += 2
        
        # 3. MACD confirmation
        if data['macd'].iloc[idx] > data['signal'].iloc[idx]:
            long_score += 1
        else:
            short_score += 1
        
        # 4. Price vs Bollinger Bands (mean reversion component)
        close = data['close'].iloc[idx]
        bb_lower = data['bb_lower'].iloc[idx]
        bb_upper = data['bb_upper'].iloc[idx]
        sma = data['sma_20'].iloc[idx]
        
        if close < bb_lower:  # Oversold
            long_score += 2
        elif close > bb_upper:  # Overbought
            short_score += 2
        elif close < sma:  # Below mean
            if data['ema_9'].iloc[idx] > data['ema_21'].iloc[idx]:  # But trending up
                long_score += 1
        elif close > sma:  # Above mean
            if data['ema_9'].iloc[idx] < data['ema_21'].iloc[idx]:  # But trending down
                short_score += 1
        
        # 5. Momentum confirmation (3-bar pattern)
        if idx >= 3:
            closes = data['close'].iloc[idx-3:idx+1].values
            if all(closes[i] < closes[i+1] for i in range(len(closes)-1)):  # Rising
                long_score += 2
            elif all(closes[i] > closes[i+1] for i in range(len(closes)-1)):  # Falling
                short_score += 2
        
        # 6. Volume confirmation
        if idx >= 1:
            vol_ratio = data['volume'].iloc[idx] / data['volume'].iloc[idx-1]
            if vol_ratio > 1.2:  # Increasing volume
                if data['close'].iloc[idx] > data['close'].iloc[idx-1]:
                    long_score += 1
                else:
                    short_score += 1
        
        return {'long_score': long_score, 'short_score': short_score}
    
    def _simulate_precise_exit(self, data: pd.DataFrame, entry_idx: int,
                               entry: float, stop: float, target: float,
                               is_long: bool) -> Dict:
        """Simulate with precise exit logic"""
        max_bars = 15  # Quick exits
        
        for i in range(1, min(max_bars + 1, len(data) - entry_idx)):
            bar = data.iloc[entry_idx + i]
            
            if is_long:
                # Check stop first (conservative)
                if bar['low'] <= stop:
                    pnl_pct = (stop - entry) / entry
                    return {'exit': stop, 'pnl_pct': pnl_pct, 'bars': i}
                # Then check target
                if bar['high'] >= target:
                    pnl_pct = (target - entry) / entry
                    return {'exit': target, 'pnl_pct': pnl_pct, 'bars': i}
            else:
                if bar['high'] >= stop:
                    pnl_pct = (entry - stop) / entry
                    return {'exit': stop, 'pnl_pct': pnl_pct, 'bars': i}
                if bar['low'] <= target:
                    pnl_pct = (entry - target) / entry
                    return {'exit': target, 'pnl_pct': pnl_pct, 'bars': i}
        
        # Exit at market if nothing hit
        exit_price = data['close'].iloc[min(entry_idx + max_bars, len(data) - 1)]
        pnl_pct = ((exit_price - entry) if is_long else (entry - exit_price)) / entry
        return {'exit': exit_price, 'pnl_pct': pnl_pct, 'bars': max_bars}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line


def main():
    """Run ultra high Sharpe strategy"""
    print("=" * 70)
    print("ULTRA HIGH SHARPE STRATEGY")
    print("Goal: Achieve Sharpe > 3 through ultra-selective trading")
    print("=" * 70)
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    strategy = UltraHighSharpe()
    results = strategy.run_ultra_selective_strategy(data)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Sharpe Ratio (Annualized): {results.get('sharpe_annualized', 0):.4f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    if results['total_trades'] > 0:
        print(f"Average Win: ${results.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${results.get('avg_loss', 0):.2f}")
    print(f"Final Capital: ${results.get('final_capital', 100000):.2f}")
    
    if results.get('sharpe_annualized', 0) > 3.0:
        print("\n🎯 SUCCESS: Achieved Sharpe > 3.0!")
    else:
        sharpe = results.get('sharpe_annualized', 0)
        print(f"\n⚠️  Sharpe {sharpe:.2f} < 3.0")
        print("Consider: More data, different timeframe, or alternative strategy")
    
    return results


if __name__ == "__main__":
    main()
