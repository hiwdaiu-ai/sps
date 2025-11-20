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
    
    def __init__(self, use_ml: bool = True):
        self.model = None
        self.use_ml = use_ml
        self.is_trained = False
        self.regimes = ['trending', 'ranging', 'high_volatility', 'low_volatility']
        self.regime_to_idx = {regime: idx for idx, regime in enumerate(self.regimes)}
        self.idx_to_regime = {idx: regime for regime, idx in self.regime_to_idx.items()}
    
    def extract_regime_features(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Extract features for regime classification
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index to extract features at
            
        Returns:
            Dictionary of regime features
        """
        features = {}
        
        # Calculate technical indicators
        df = df.copy()
        df['atr'] = self._calculate_atr(df)
        df['ema_fast'] = df['close'].ewm(span=12).mean()
        df['ema_slow'] = df['close'].ewm(span=26).mean()
        df['ema_long'] = df['close'].ewm(span=50).mean()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Volatility features
        recent_atr = df.iloc[idx]['atr']
        avg_atr = df['atr'].iloc[max(0, idx-50):idx].mean()
        features['volatility_ratio'] = recent_atr / avg_atr if avg_atr > 0 else 1.0
        features['atr_normalized'] = recent_atr / df.iloc[idx]['close']
        
        # Calculate rolling volatility
        if idx >= 20:
            returns = df['close'].pct_change()
            features['volatility_std'] = returns.iloc[idx-20:idx].std()
        else:
            features['volatility_std'] = 0.02
        
        # Trend features
        ema_diff = df.iloc[idx]['ema_fast'] - df.iloc[idx]['ema_slow']
        features['trend_strength'] = ema_diff / df.iloc[idx]['close']
        features['ema_slope_fast'] = (df.iloc[idx]['ema_fast'] - df.iloc[max(0, idx-5)]['ema_fast']) / df.iloc[idx]['close']
        features['ema_slope_slow'] = (df.iloc[idx]['ema_slow'] - df.iloc[max(0, idx-5)]['ema_slow']) / df.iloc[idx]['close']
        
        # Price action features
        if idx >= 10:
            high_range = df.iloc[idx-10:idx]['high'].max() - df.iloc[idx-10:idx]['low'].min()
            features['price_range'] = high_range / df.iloc[idx]['close']
        else:
            features['price_range'] = 0.05
        
        # Volume features
        features['volume_ratio'] = df.iloc[idx]['volume'] / df.iloc[idx]['volume_ma'] if df.iloc[idx]['volume_ma'] > 0 else 1.0
        
        # ADX-like directional movement
        if idx >= 14:
            plus_dm = (df['high'].diff()).clip(lower=0)
            minus_dm = (-df['low'].diff()).clip(lower=0)
            features['directional_strength'] = abs(plus_dm.iloc[idx-14:idx].mean() - minus_dm.iloc[idx-14:idx].mean()) / recent_atr if recent_atr > 0 else 0
        else:
            features['directional_strength'] = 0.5
        
        return features
    
    def train(self, df: pd.DataFrame, lookback: int = 200):
        """
        Train the regime detector using ML
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to use for training
        """
        if not self.use_ml:
            return
        
        print("Training regime detector...")
        
        # Generate training data
        features_list = []
        labels = []
        
        start_idx = max(100, len(df) - lookback) if lookback else 100
        
        for idx in range(start_idx, len(df)):
            # Extract features
            features = self.extract_regime_features(df, idx)
            features_list.append(features)
            
            # Generate label using rule-based classification
            label = self._rule_based_classify(features)
            labels.append(self.regime_to_idx[label])
        
        if len(features_list) < 20:
            print("Warning: Not enough data for regime training")
            return
        
        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        # Train LightGBM model
        params = {
            'objective': 'multiclass',
            'num_class': len(self.regimes),
            'metric': 'multi_logloss',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        self.is_trained = True
        
        # Print regime distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Trained on {len(y)} samples")
        for regime_idx, count in zip(unique, counts):
            print(f"  {self.idx_to_regime[regime_idx]}: {count} samples")
    
    def detect_regime(self, df: pd.DataFrame, idx: int = -1) -> str:
        """
        Detect current market regime using ML or rules
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index to detect regime at (default: latest)
            
        Returns:
            Regime string
        """
        if len(df) < 50:
            return 'ranging'
        
        # Extract features
        features = self.extract_regime_features(df, idx)
        
        # Use ML model if trained
        if self.use_ml and self.is_trained:
            feature_df = pd.DataFrame([features])
            predictions = self.model.predict(feature_df)
            regime_idx = int(np.argmax(predictions[0]))
            return self.idx_to_regime[regime_idx]
        else:
            # Fall back to rule-based
            return self._rule_based_classify(features)
    
    def _rule_based_classify(self, features: Dict[str, float]) -> str:
        """Rule-based regime classification"""
        volatility_ratio = features.get('volatility_ratio', 1.0)
        trend_strength = abs(features.get('trend_strength', 0.0))
        
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
