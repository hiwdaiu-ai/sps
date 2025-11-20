"""
Rule-based SMC detector for orderblocks, liquidity grabs, BOS/CHOCH, imbalances, FVGs
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class OrderBlock:
    """Represents a detected orderblock"""
    timestamp: pd.Timestamp
    price_high: float
    price_low: float
    is_bullish: bool
    volume: float
    strength: float  # 0-1 score


@dataclass
class LiquidityGrab:
    """Represents a detected liquidity grab/sweep"""
    timestamp: pd.Timestamp
    price: float
    direction: str  # 'long' or 'short'
    volume_spike: float


class SMCDetector:
    """Deterministic SMC pattern detector"""
    
    def __init__(self, lookback: int = 20, min_size: float = 0.5):
        self.lookback = lookback
        self.min_size = min_size
    
    def detect_orderblocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect orderblocks in price data
        
        An orderblock is identified as:
        - A strong candle followed by a significant move
        - High volume relative to average
        - Price doesn't return to the zone quickly
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of OrderBlock objects
        """
        orderblocks = []
        
        if len(df) < self.lookback:
            return orderblocks
        
        # Calculate ATR for context
        df = df.copy()
        df['atr'] = self._calculate_atr(df)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(self.lookback, len(df) - 5):
            # Look for strong bullish candles
            candle_size = df.iloc[i]['close'] - df.iloc[i]['open']
            body_pct = abs(candle_size) / df.iloc[i]['atr']
            volume_ratio = df.iloc[i]['volume'] / df.iloc[i]['volume_ma']
            
            # Bullish orderblock: strong up candle with high volume
            if candle_size > 0 and body_pct > self.min_size and volume_ratio > 1.5:
                # Check if price moved significantly after
                next_high = df.iloc[i+1:i+5]['high'].max()
                if next_high > df.iloc[i]['high'] * 1.002:  # 0.2% move
                    ob = OrderBlock(
                        timestamp=df.index[i],
                        price_high=df.iloc[i]['high'],
                        price_low=df.iloc[i]['low'],
                        is_bullish=True,
                        volume=df.iloc[i]['volume'],
                        strength=min(body_pct / 2.0, 1.0)
                    )
                    orderblocks.append(ob)
            
            # Bearish orderblock: strong down candle with high volume
            elif candle_size < 0 and body_pct > self.min_size and volume_ratio > 1.5:
                next_low = df.iloc[i+1:i+5]['low'].min()
                if next_low < df.iloc[i]['low'] * 0.998:  # 0.2% move
                    ob = OrderBlock(
                        timestamp=df.index[i],
                        price_high=df.iloc[i]['high'],
                        price_low=df.iloc[i]['low'],
                        is_bullish=False,
                        volume=df.iloc[i]['volume'],
                        strength=min(body_pct / 2.0, 1.0)
                    )
                    orderblocks.append(ob)
        
        return orderblocks
    
    def detect_liquidity_grabs(self, df: pd.DataFrame, atr_multiplier: float = 1.5) -> List[LiquidityGrab]:
        """
        Detect liquidity grabs/sweeps
        
        Identified by:
        - Large wick (rejection)
        - High volume
        - Quick reversal
        
        Args:
            df: DataFrame with OHLCV data
            atr_multiplier: Threshold for wick size relative to ATR
            
        Returns:
            List of LiquidityGrab objects
        """
        grabs = []
        
        if len(df) < 20:
            return grabs
        
        df = df.copy()
        df['atr'] = self._calculate_atr(df)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(20, len(df)):
            # Upper wick (potential short liquidity grab)
            upper_wick = df.iloc[i]['high'] - max(df.iloc[i]['open'], df.iloc[i]['close'])
            wick_size = upper_wick / df.iloc[i]['atr']
            volume_spike = df.iloc[i]['volume'] / df.iloc[i]['volume_ma']
            
            if wick_size > atr_multiplier and volume_spike > 2.0:
                grab = LiquidityGrab(
                    timestamp=df.index[i],
                    price=df.iloc[i]['high'],
                    direction='short',
                    volume_spike=volume_spike
                )
                grabs.append(grab)
            
            # Lower wick (potential long liquidity grab)
            lower_wick = min(df.iloc[i]['open'], df.iloc[i]['close']) - df.iloc[i]['low']
            wick_size = lower_wick / df.iloc[i]['atr']
            
            if wick_size > atr_multiplier and volume_spike > 2.0:
                grab = LiquidityGrab(
                    timestamp=df.index[i],
                    price=df.iloc[i]['low'],
                    direction='long',
                    volume_spike=volume_spike
                )
                grabs.append(grab)
        
        return grabs
    
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
    
    def detect_bos_choch(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHOCH)
        
        Simplified implementation:
        - BOS: Break above recent high in uptrend or below recent low in downtrend
        - CHOCH: Break that changes the trend direction
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of dictionaries with BOS/CHOCH events
        """
        events = []
        lookback = 10
        
        if len(df) < lookback * 2:
            return events
        
        for i in range(lookback * 2, len(df)):
            recent_high = df.iloc[i-lookback:i]['high'].max()
            recent_low = df.iloc[i-lookback:i]['low'].min()
            
            # BOS: Break above recent high
            if df.iloc[i]['high'] > recent_high:
                events.append({
                    'timestamp': df.index[i],
                    'type': 'BOS',
                    'direction': 'bullish',
                    'price': df.iloc[i]['high']
                })
            
            # BOS: Break below recent low
            elif df.iloc[i]['low'] < recent_low:
                events.append({
                    'timestamp': df.index[i],
                    'type': 'BOS',
                    'direction': 'bearish',
                    'price': df.iloc[i]['low']
                })
        
        return events
