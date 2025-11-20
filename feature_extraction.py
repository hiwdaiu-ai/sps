"""
Feature engineering pipeline for SMC candidates
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from smc_detector import OrderBlock, LiquidityGrab


class FeatureExtractor:
    """Extract numeric features from SMC candidates"""
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
    
    def extract_features(self, candidate: OrderBlock, df: pd.DataFrame, 
                        current_idx: int) -> Dict[str, float]:
        """
        Extract features for an orderblock candidate
        
        Args:
            candidate: OrderBlock object
            df: DataFrame with OHLCV data
            current_idx: Current bar index for distance calculation
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Calculate technical indicators
        df = df.copy()
        df['atr'] = self._calculate_atr(df)
        df['ema_fast'] = df['close'].ewm(span=12).mean()
        df['ema_slow'] = df['close'].ewm(span=26).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        current_price = df.iloc[current_idx]['close']
        current_atr = df.iloc[current_idx]['atr']
        
        # Distance features
        zone_mid = (candidate.price_high + candidate.price_low) / 2
        features['distance_to_zone'] = (current_price - zone_mid) / current_atr
        features['zone_size'] = (candidate.price_high - candidate.price_low) / current_atr
        
        # Volume features
        features['candidate_volume'] = candidate.volume
        features['volume_ratio'] = candidate.volume / df['volume'].mean()
        
        # Candle features
        candle = df.iloc[current_idx]
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        features['body_ratio'] = body / total_range if total_range > 0 else 0
        
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        features['upper_wick_ratio'] = upper_wick / total_range if total_range > 0 else 0
        features['lower_wick_ratio'] = lower_wick / total_range if total_range > 0 else 0
        
        # Momentum features
        features['ema_fast'] = df.iloc[current_idx]['ema_fast']
        features['ema_slow'] = df.iloc[current_idx]['ema_slow']
        features['ema_diff'] = (features['ema_fast'] - features['ema_slow']) / current_price
        
        # Volatility
        features['atr'] = current_atr
        features['atr_normalized'] = current_atr / current_price
        
        # RSI
        features['rsi'] = df.iloc[current_idx]['rsi']
        
        # Orderblock strength
        features['ob_strength'] = candidate.strength
        features['ob_is_bullish'] = 1.0 if candidate.is_bullish else 0.0
        
        # Time features
        features['bars_since_ob'] = current_idx - df.index.get_loc(candidate.timestamp)
        
        return features
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_feature_matrix(self, candidates: List[OrderBlock], 
                             df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix for multiple candidates
        
        Args:
            candidates: List of OrderBlock objects
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features for each candidate
        """
        feature_list = []
        
        for i, candidate in enumerate(candidates):
            # Get index of candidate
            try:
                candidate_idx = df.index.get_loc(candidate.timestamp)
                # Use a future bar as "current" for feature extraction
                if candidate_idx + 5 < len(df):
                    current_idx = candidate_idx + 5
                    features = self.extract_features(candidate, df, current_idx)
                    feature_list.append(features)
            except KeyError:
                continue
        
        if not feature_list:
            return pd.DataFrame()
        
        return pd.DataFrame(feature_list)
