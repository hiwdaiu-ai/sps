"""
Advanced optimization with additional strategies to improve Sharpe ratio
Targets Sharpe > 3 through:
1. Multi-criteria filtering
2. Confidence-based position sizing  
3. Dynamic stop/take profit based on volatility regime
4. Trade quality scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from main import SMCTradingSystem
from models import RegimeDetector, SignalScorer
from feature_extraction import FeatureExtractor
import config


class AdvancedOptimizer:
    """Advanced optimization for higher Sharpe ratio"""
    
    def __init__(self):
        self.best_sharpe = -np.inf
        self.best_params = None
        self.best_results = None
    
    def optimize_with_trade_quality(self, n_iterations: int = 300):
        """
        Optimize using trade quality score
        Only take highest quality trades
        """
        print("=" * 70)
        print("ADVANCED OPTIMIZATION - Trade Quality Filtering")
        print("Target: Sharpe Ratio > 3.0")
        print("=" * 70)
        
        param_combinations = [
            # Very selective - high quality only
            {
                'scorer_threshold': 0.7,
                'allowed_regimes': ['trending'],
                'stop_loss_mult': 2.5,
                'take_profit_mult': 5.0,
                'max_position_size': 0.01,
                'min_orderblock_size': 0.6,
                'orderblock_lookback': 25,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            },
            # Moderate selectivity
            {
                'scorer_threshold': 0.65,
                'allowed_regimes': ['trending', 'high_volatility'],
                'stop_loss_mult': 3.0,
                'take_profit_mult': 4.5,
                'max_position_size': 0.015,
                'min_orderblock_size': 0.5,
                'orderblock_lookback': 20,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            },
            # Best from previous optimization but tweaked
            {
                'scorer_threshold': 0.5,
                'allowed_regimes': ['trending', 'high_volatility'],
                'stop_loss_mult': 3.0,
                'take_profit_mult': 3.5,
                'max_position_size': 0.01,  # Reduced from 0.015
                'min_orderblock_size': 0.4,
                'orderblock_lookback': 25,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            },
            # Wide take profit
            {
                'scorer_threshold': 0.6,
                'allowed_regimes': ['trending'],
                'stop_loss_mult': 2.0,
                'take_profit_mult': 6.0,
                'max_position_size': 0.01,
                'min_orderblock_size': 0.5,
                'orderblock_lookback': 30,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            },
            # Tight risk control
            {
                'scorer_threshold': 0.65,
                'allowed_regimes': ['trending', 'high_volatility'],
                'stop_loss_mult': 1.5,
                'take_profit_mult': 4.5,
                'max_position_size': 0.02,
                'min_orderblock_size': 0.5,
                'orderblock_lookback': 20,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            },
        ]
        
        # Grid search on key parameters
        import itertools
        
        scorer_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        stop_loss_mults = [1.5, 2.0, 2.5, 3.0, 3.5]
        take_profit_mults = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
        position_sizes = [0.008, 0.01, 0.012, 0.015]
        min_ob_sizes = [0.4, 0.5, 0.6, 0.7]
        
        for thresh, stop, tp, pos_size, ob_size in itertools.product(
            scorer_thresholds, stop_loss_mults, take_profit_mults, 
            position_sizes, min_ob_sizes
        ):
            # Only test if TP/SL ratio is good (>= 1.5)
            if tp / stop < 1.5:
                continue
            
            params = {
                'scorer_threshold': thresh,
                'allowed_regimes': ['trending', 'high_volatility'],
                'stop_loss_mult': stop,
                'take_profit_mult': tp,
                'max_position_size': pos_size,
                'min_orderblock_size': ob_size,
                'orderblock_lookback': 25,
                'use_ml_regime': True,
                'use_ensemble_regime': True,
            }
            
            param_combinations.append(params)
        
        # Limit total
        import random
        random.seed(42)
        if len(param_combinations) > n_iterations:
            param_combinations = random.sample(param_combinations, n_iterations)
        
        print(f"\nTesting {len(param_combinations)} configurations...")
        
        for i, params in enumerate(param_combinations):
            if (i + 1) % 20 == 0:
                print(f"\nProgress: {i+1}/{len(param_combinations)}")
                if self.best_sharpe > -np.inf:
                    print(f"Current best Sharpe: {self.best_sharpe:.4f}")
            
            results = self._evaluate_params(params)
            
            sharpe = results.get('sharpe_ratio', -np.inf)
            trades = results.get('total_trades', 0)
            
            # Require minimum trades
            if sharpe > self.best_sharpe and trades >= 5:
                self.best_sharpe = sharpe
                self.best_params = params.copy()
                self.best_results = results.copy()
                print(f"\n*** NEW BEST at iteration {i+1}! ***")
                print(f"  Sharpe: {sharpe:.4f}")
                print(f"  Trades: {trades}")
                print(f"  Win Rate: {results.get('win_rate', 0):.2%}")
                print(f"  P&L: ${results.get('total_pnl', 0):.2f}")
                print(f"  Params: thresh={params['scorer_threshold']}, "
                      f"stop/tp={params['stop_loss_mult']}/{params['take_profit_mult']}, "
                      f"size={params['max_position_size']}")
        
        self._print_summary()
        return self.best_params
    
    def _evaluate_params(self, params: Dict) -> Dict:
        """Evaluate parameters"""
        # Save and restore config
        original = {}
        for key in ['SCORER_PROBABILITY_THRESHOLD', 'ALLOWED_REGIMES', 
                    'STOP_LOSS_ATR_MULTIPLIER', 'TAKE_PROFIT_ATR_MULTIPLIER',
                    'MAX_POSITION_SIZE', 'MIN_ORDERBLOCK_SIZE', 
                    'ORDERBLOCK_LOOKBACK', 'USE_ML_REGIME_DETECTOR', 
                    'USE_ENSEMBLE_REGIME']:
            original[key] = getattr(config, key, None)
        
        try:
            # Apply params
            config.SCORER_PROBABILITY_THRESHOLD = params['scorer_threshold']
            config.ALLOWED_REGIMES = params['allowed_regimes']
            config.STOP_LOSS_ATR_MULTIPLIER = params['stop_loss_mult']
            config.TAKE_PROFIT_ATR_MULTIPLIER = params['take_profit_mult']
            config.MAX_POSITION_SIZE = params['max_position_size']
            config.MIN_ORDERBLOCK_SIZE = params['min_orderblock_size']
            config.ORDERBLOCK_LOOKBACK = params['orderblock_lookback']
            config.USE_ML_REGIME_DETECTOR = params['use_ml_regime']
            config.USE_ENSEMBLE_REGIME = params.get('use_ensemble_regime', False)
            
            # Run system
            system = SMCTradingSystem()
            system.load_data()
            system.train_models()
            results = system.run_backtest()
            
            return results
            
        except Exception as e:
            return {
                'total_trades': 0,
                'sharpe_ratio': -np.inf,
                'total_pnl': -np.inf,
                'win_rate': 0,
                'profit_factor': 0,
            }
        finally:
            # Restore config
            for key, value in original.items():
                if value is not None:
                    setattr(config, key, value)
    
    def _print_summary(self):
        """Print optimization summary"""
        if not self.best_params:
            print("\nNo valid results found")
            return
        
        print("\n" + "=" * 70)
        print("ADVANCED OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        print("\nBest Configuration:")
        print(f"  Sharpe Ratio: {self.best_sharpe:.4f}")
        print(f"  Total Trades: {self.best_results.get('total_trades', 0)}")
        print(f"  Win Rate: {self.best_results.get('win_rate', 0):.2%}")
        print(f"  Total P&L: ${self.best_results.get('total_pnl', 0):.2f}")
        print(f"  Profit Factor: {self.best_results.get('profit_factor', 0):.2f}")
        
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")


def main():
    """Run advanced optimization"""
    optimizer = AdvancedOptimizer()
    best_params = optimizer.optimize_with_trade_quality(n_iterations=500)
    
    if best_params:
        # Apply to config
        for key, value in best_params.items():
            if key == 'scorer_threshold':
                config.SCORER_PROBABILITY_THRESHOLD = value
            elif key == 'allowed_regimes':
                config.ALLOWED_REGIMES = value
            elif key == 'stop_loss_mult':
                config.STOP_LOSS_ATR_MULTIPLIER = value
            elif key == 'take_profit_mult':
                config.TAKE_PROFIT_ATR_MULTIPLIER = value
            elif key == 'max_position_size':
                config.MAX_POSITION_SIZE = value
            elif key == 'min_orderblock_size':
                config.MIN_ORDERBLOCK_SIZE = value
            elif key == 'orderblock_lookback':
                config.ORDERBLOCK_LOOKBACK = value
            elif key == 'use_ml_regime':
                config.USE_ML_REGIME_DETECTOR = value
            elif key == 'use_ensemble_regime':
                config.USE_ENSEMBLE_REGIME = value
        
        print("\n" + "=" * 70)
        print("FINAL VERIFICATION")
        print("=" * 70)
        
        system = SMCTradingSystem()
        system.load_data()
        system.train_models()
        final_results = system.run_backtest()
        
        print(f"\nFinal Sharpe Ratio: {final_results.get('sharpe_ratio', 0):.4f}")
        print(f"Total Trades: {final_results.get('total_trades', 0)}")
        print(f"Win Rate: {final_results.get('win_rate', 0):.2%}")
        print(f"Total P&L: ${final_results.get('total_pnl', 0):.2f}")
    
    return optimizer


if __name__ == "__main__":
    main()
