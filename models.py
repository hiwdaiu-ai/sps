"""
Regime detection and signal scoring models
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV


class RegimeDetector:
    """Detect market regime (trending, ranging, high-volatility, low-liquidity)"""
    
    def __init__(self):
        self.model = None
        self.regimes = ['trending', 'ranging', 'high_volatility', 'low_volatility']
    
    def detect_regime(self, df: pd.DataFrame, idx: int = -1) -> str:
        """
        Detect current market regime using simple rules
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index to detect regime at (default: latest)
            
        Returns:
            Regime string
        """
        if len(df) < 50:
            return 'ranging'
        
        # Calculate indicators
        df = df.copy()
        df['atr'] = self._calculate_atr(df)
        df['ema_fast'] = df['close'].ewm(span=12).mean()
        df['ema_slow'] = df['close'].ewm(span=26).mean()
        
        # Volatility metric
        recent_atr = df.iloc[idx]['atr']
        avg_atr = df['atr'].iloc[-50:].mean()
        volatility_ratio = recent_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Trend strength
        ema_diff = abs(df.iloc[idx]['ema_fast'] - df.iloc[idx]['ema_slow'])
        trend_strength = ema_diff / df.iloc[idx]['close']
        
        # Classify regime
        if volatility_ratio > 1.5:
            return 'high_volatility'
        elif volatility_ratio < 0.7:
            return 'low_volatility'
        elif trend_strength > 0.02:
            return 'trending'
        else:
            return 'ranging'
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr


class SignalScorer:
    """Score SMC candidates using LightGBM with calibrated probabilities"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: np.ndarray):
        """
        Train the scoring model
        
        Args:
            X: Feature matrix
            y: Binary labels (1=profitable, 0=unprofitable)
        """
        # Train LightGBM model
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        self.is_trained = True
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of success for candidates
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities
        """
        if not self.is_trained:
            # Return neutral probabilities if not trained
            return np.full(len(X), 0.5)
        
        return self.model.predict(X)
    
    def generate_training_labels(self, df: pd.DataFrame, candidates: list, 
                                 horizon: int = 10) -> np.ndarray:
        """
        Generate outcome-based labels for training
        
        Args:
            df: DataFrame with OHLCV data
            candidates: List of OrderBlock candidates
            horizon: Number of bars to look ahead for outcome
            
        Returns:
            Array of binary labels
        """
        labels = []
        
        for candidate in candidates:
            try:
                candidate_idx = df.index.get_loc(candidate.timestamp)
                
                # Look ahead to see if price reached the zone profitably
                if candidate_idx + horizon < len(df):
                    entry_price = candidate.price_low if candidate.is_bullish else candidate.price_high
                    future_prices = df.iloc[candidate_idx:candidate_idx+horizon]
                    
                    if candidate.is_bullish:
                        # For bullish: check if price went up
                        max_price = future_prices['high'].max()
                        profit_pct = (max_price - entry_price) / entry_price
                    else:
                        # For bearish: check if price went down
                        min_price = future_prices['low'].min()
                        profit_pct = (entry_price - min_price) / entry_price
                    
                    # Label as 1 if profit > 1%, 0 otherwise
                    label = 1 if profit_pct > 0.01 else 0
                    labels.append(label)
            except (KeyError, IndexError):
                continue
        
        return np.array(labels)
