"""
Hyperparameter optimization for SMC Trading System
Goal: Maximize Sharpe ratio (target > 3) with good number of trades
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import itertools
from main import SMCTradingSystem
import config


class HyperparameterOptimizer:
    """Optimize hyperparameters to maximize Sharpe ratio"""
    
    def __init__(self, data_filepath: Optional[str] = None):
        self.data_filepath = data_filepath
        self.best_params = None
        self.best_sharpe = -np.inf
        self.best_results = None
        self.optimization_history = []
    
    def create_param_grid(self) -> Dict[str, List]:
        """
        Create parameter grid for hyperparameter search
        Focus on parameters that affect trading decisions and risk
        """
        param_grid = {
            # Scoring threshold - critical for trade selection
            'scorer_threshold': [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            
            # Regime filtering - combine multiple regimes
            'allowed_regimes': [
                ['trending'],
                ['trending', 'high_volatility'],
                ['trending', 'ranging'],
                ['trending', 'high_volatility', 'ranging'],
                ['trending', 'high_volatility', 'low_volatility'],
                ['trending', 'ranging', 'high_volatility', 'low_volatility'],  # All regimes
            ],
            
            # Risk management - key for Sharpe optimization
            'stop_loss_mult': [1.5, 2.0, 2.5, 3.0, 3.5],
            'take_profit_mult': [2.0, 2.5, 3.0, 3.5, 4.0, 5.0],
            'max_position_size': [0.01, 0.015, 0.02, 0.025, 0.03],
            
            # SMC detection sensitivity
            'min_orderblock_size': [0.3, 0.4, 0.5, 0.6, 0.7],
            'orderblock_lookback': [15, 20, 25, 30],
            
            # ML vs Rule-based vs Ensemble regime detection
            'use_ml_regime': [True, False],
            'use_ensemble_regime': [True, False],
        }
        
        return param_grid
    
    def evaluate_params(self, params: Dict) -> Dict:
        """
        Evaluate a single parameter configuration
        
        Returns:
            Dictionary with performance metrics
        """
        # Temporarily override config
        original_config = {}
        
        # Save original config
        original_config['SCORER_PROBABILITY_THRESHOLD'] = config.SCORER_PROBABILITY_THRESHOLD
        original_config['ALLOWED_REGIMES'] = config.ALLOWED_REGIMES
        original_config['STOP_LOSS_ATR_MULTIPLIER'] = config.STOP_LOSS_ATR_MULTIPLIER
        original_config['TAKE_PROFIT_ATR_MULTIPLIER'] = config.TAKE_PROFIT_ATR_MULTIPLIER
        original_config['MAX_POSITION_SIZE'] = config.MAX_POSITION_SIZE
        original_config['MIN_ORDERBLOCK_SIZE'] = config.MIN_ORDERBLOCK_SIZE
        original_config['ORDERBLOCK_LOOKBACK'] = config.ORDERBLOCK_LOOKBACK
        original_config['USE_ML_REGIME_DETECTOR'] = config.USE_ML_REGIME_DETECTOR
        original_config['USE_ENSEMBLE_REGIME'] = getattr(config, 'USE_ENSEMBLE_REGIME', False)
        
        # Apply new params
        config.SCORER_PROBABILITY_THRESHOLD = params['scorer_threshold']
        config.ALLOWED_REGIMES = params['allowed_regimes']
        config.STOP_LOSS_ATR_MULTIPLIER = params['stop_loss_mult']
        config.TAKE_PROFIT_ATR_MULTIPLIER = params['take_profit_mult']
        config.MAX_POSITION_SIZE = params['max_position_size']
        config.MIN_ORDERBLOCK_SIZE = params['min_orderblock_size']
        config.ORDERBLOCK_LOOKBACK = params['orderblock_lookback']
        config.USE_ML_REGIME_DETECTOR = params['use_ml_regime']
        config.USE_ENSEMBLE_REGIME = params.get('use_ensemble_regime', False)
        
        try:
            # Create and run system
            system = SMCTradingSystem()
            system.load_data(self.data_filepath)
            system.train_models()
            results = system.run_backtest()
            
            # Restore config
            for key, value in original_config.items():
                setattr(config, key, value)
            
            return results
            
        except Exception as e:
            # Restore config on error
            for key, value in original_config.items():
                setattr(config, key, value)
            
            return {
                'total_trades': 0,
                'sharpe_ratio': -np.inf,
                'total_pnl': -np.inf,
                'win_rate': 0,
                'profit_factor': 0,
                'error': str(e)
            }
    
    def grid_search(self, max_iterations: Optional[int] = None) -> Dict:
        """
        Perform grid search over parameter space
        
        Args:
            max_iterations: Maximum number of parameter combinations to try (None = all)
        
        Returns:
            Best parameters and results
        """
        param_grid = self.create_param_grid()
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combinations = list(itertools.product(*values))
        
        print(f"Total parameter combinations: {len(all_combinations)}")
        
        if max_iterations:
            # Random sample if too many
            import random
            random.seed(42)
            all_combinations = random.sample(all_combinations, min(max_iterations, len(all_combinations)))
            print(f"Testing {len(all_combinations)} random combinations")
        
        # Test each combination
        for i, combo in enumerate(all_combinations):
            params = dict(zip(keys, combo))
            
            print(f"\n[{i+1}/{len(all_combinations)}] Testing parameters:")
            print(f"  Scorer threshold: {params['scorer_threshold']}")
            print(f"  Regimes: {params['allowed_regimes']}")
            print(f"  Stop/TP mult: {params['stop_loss_mult']}/{params['take_profit_mult']}")
            print(f"  Position size: {params['max_position_size']}")
            print(f"  ML regime: {params['use_ml_regime']}")
            
            results = self.evaluate_params(params)
            
            # Store in history
            self.optimization_history.append({
                'params': params.copy(),
                'results': results.copy()
            })
            
            sharpe = results.get('sharpe_ratio', -np.inf)
            trades = results.get('total_trades', 0)
            pnl = results.get('total_pnl', 0)
            
            print(f"  Results: Sharpe={sharpe:.3f}, Trades={trades}, P&L=${pnl:.2f}")
            
            # Update best if this is better
            # Require minimum 5 trades to avoid overfitting
            if sharpe > self.best_sharpe and trades >= 5:
                self.best_sharpe = sharpe
                self.best_params = params.copy()
                self.best_results = results.copy()
                print(f"  *** NEW BEST! Sharpe={sharpe:.3f} ***")
        
        return {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'best_results': self.best_results,
            'history': self.optimization_history
        }
    
    def smart_search(self, n_iterations: int = 100) -> Dict:
        """
        Smart search focusing on promising parameter regions
        Uses bayesian-like approach without external dependencies
        """
        param_grid = self.create_param_grid()
        
        print(f"Starting smart search with {n_iterations} iterations")
        
        # Stage 1: Quick grid search on key parameters (30% of budget)
        stage1_iterations = int(n_iterations * 0.3)
        print(f"\n=== Stage 1: Quick exploration ({stage1_iterations} iterations) ===")
        
        # Focus on most impactful parameters
        key_params = {
            'scorer_threshold': param_grid['scorer_threshold'],
            'allowed_regimes': param_grid['allowed_regimes'][:3],  # Top 3 regime combos
            'stop_loss_mult': [1.5, 2.0, 2.5, 3.0],
            'take_profit_mult': [2.5, 3.0, 4.0, 5.0],
            'max_position_size': [0.015, 0.02, 0.025],
            'min_orderblock_size': [0.4, 0.5, 0.6],
            'orderblock_lookback': [20, 25],
            'use_ml_regime': [True, False],
            'use_ensemble_regime': [True, False],
        }
        
        import random
        random.seed(42)
        
        for i in range(stage1_iterations):
            params = {k: random.choice(v) for k, v in key_params.items()}
            
            print(f"\n[Stage 1: {i+1}/{stage1_iterations}]")
            print(f"  Params: thresh={params['scorer_threshold']}, "
                  f"regimes={len(params['allowed_regimes'])}, "
                  f"stop/tp={params['stop_loss_mult']}/{params['take_profit_mult']}")
            
            results = self.evaluate_params(params)
            
            self.optimization_history.append({
                'params': params.copy(),
                'results': results.copy(),
                'stage': 1
            })
            
            sharpe = results.get('sharpe_ratio', -np.inf)
            trades = results.get('total_trades', 0)
            
            print(f"  Results: Sharpe={sharpe:.3f}, Trades={trades}")
            
            if sharpe > self.best_sharpe and trades >= 5:
                self.best_sharpe = sharpe
                self.best_params = params.copy()
                self.best_results = results.copy()
                print(f"  *** NEW BEST! ***")
        
        # Stage 2: Refine around best (70% of budget)
        stage2_iterations = n_iterations - stage1_iterations
        print(f"\n=== Stage 2: Refinement ({stage2_iterations} iterations) ===")
        print(f"Best from Stage 1: Sharpe={self.best_sharpe:.3f}")
        
        if self.best_params:
            # Refine around best parameters
            for i in range(stage2_iterations):
                # Perturb best parameters
                params = self.best_params.copy()
                
                # Randomly modify 1-3 parameters
                n_changes = random.randint(1, 3)
                params_to_change = random.sample(list(param_grid.keys()), n_changes)
                
                for param_name in params_to_change:
                    params[param_name] = random.choice(param_grid[param_name])
                
                print(f"\n[Stage 2: {i+1}/{stage2_iterations}]")
                
                results = self.evaluate_params(params)
                
                self.optimization_history.append({
                    'params': params.copy(),
                    'results': results.copy(),
                    'stage': 2
                })
                
                sharpe = results.get('sharpe_ratio', -np.inf)
                trades = results.get('total_trades', 0)
                
                print(f"  Results: Sharpe={sharpe:.3f}, Trades={trades}")
                
                if sharpe > self.best_sharpe and trades >= 5:
                    self.best_sharpe = sharpe
                    self.best_params = params.copy()
                    self.best_results = results.copy()
                    print(f"  *** NEW BEST! Sharpe={sharpe:.3f} ***")
        
        return {
            'best_params': self.best_params,
            'best_sharpe': self.best_sharpe,
            'best_results': self.best_results,
            'history': self.optimization_history
        }
    
    def print_summary(self):
        """Print optimization summary"""
        if not self.best_params:
            print("No optimization results available")
            return
        
        print("\n" + "=" * 70)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("=" * 70)
        
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nBest Performance:")
        print(f"  Sharpe Ratio: {self.best_sharpe:.4f}")
        print(f"  Total Trades: {self.best_results.get('total_trades', 0)}")
        print(f"  Win Rate: {self.best_results.get('win_rate', 0):.2%}")
        print(f"  Total P&L: ${self.best_results.get('total_pnl', 0):.2f}")
        print(f"  Profit Factor: {self.best_results.get('profit_factor', 0):.2f}")
        
        # Top 5 configurations
        sorted_history = sorted(
            [h for h in self.optimization_history if h['results'].get('total_trades', 0) >= 5],
            key=lambda x: x['results'].get('sharpe_ratio', -np.inf),
            reverse=True
        )[:5]
        
        print("\nTop 5 Configurations:")
        for i, entry in enumerate(sorted_history):
            sharpe = entry['results'].get('sharpe_ratio', 0)
            trades = entry['results'].get('total_trades', 0)
            pnl = entry['results'].get('total_pnl', 0)
            print(f"\n  #{i+1}: Sharpe={sharpe:.3f}, Trades={trades}, P&L=${pnl:.2f}")
            print(f"    Threshold: {entry['params']['scorer_threshold']}")
            print(f"    Regimes: {entry['params']['allowed_regimes']}")
            print(f"    Stop/TP: {entry['params']['stop_loss_mult']}/{entry['params']['take_profit_mult']}")


def apply_best_params(best_params: Dict):
    """Apply best parameters to config"""
    config.SCORER_PROBABILITY_THRESHOLD = best_params['scorer_threshold']
    config.ALLOWED_REGIMES = best_params['allowed_regimes']
    config.STOP_LOSS_ATR_MULTIPLIER = best_params['stop_loss_mult']
    config.TAKE_PROFIT_ATR_MULTIPLIER = best_params['take_profit_mult']
    config.MAX_POSITION_SIZE = best_params['max_position_size']
    config.MIN_ORDERBLOCK_SIZE = best_params['min_orderblock_size']
    config.ORDERBLOCK_LOOKBACK = best_params['orderblock_lookback']
    config.USE_ML_REGIME_DETECTOR = best_params['use_ml_regime']
    config.USE_ENSEMBLE_REGIME = best_params.get('use_ensemble_regime', False)
    
    print("\nApplied best parameters to config!")


def main():
    """Run hyperparameter optimization"""
    print("=" * 70)
    print("SMC Trading System - Hyperparameter Optimization")
    print("Goal: Sharpe Ratio > 3 with good number of trades")
    print("=" * 70)
    
    optimizer = HyperparameterOptimizer()
    
    # Use smart search for efficiency
    results = optimizer.smart_search(n_iterations=200)
    
    # Print summary
    optimizer.print_summary()
    
    # Apply best parameters
    if optimizer.best_params:
        apply_best_params(optimizer.best_params)
        
        # Final verification with best parameters
        print("\n" + "=" * 70)
        print("FINAL VERIFICATION WITH BEST PARAMETERS")
        print("=" * 70)
        
        system = SMCTradingSystem()
        system.load_data()
        system.train_models()
        final_results = system.run_backtest()
        
        print(f"\nFinal Results:")
        for metric, value in final_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    return results


if __name__ == "__main__":
    main()
