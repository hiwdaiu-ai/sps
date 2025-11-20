"""
Data ingestion module for OHLCV and tick data
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class DataIngestion:
    """Handle OHLCV data ingestion and preprocessing"""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = None
    
    def load_ohlcv(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from file or API
        
        Args:
            filepath: Path to CSV file with OHLCV data
            
        Returns:
            DataFrame with OHLCV data
        """
        if filepath:
            # Load from CSV
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            # Generate sample data for demonstration
            df = self._generate_sample_data()
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        self.data = df
        return df
    
    def _generate_sample_data(self, n_bars: int = 1000) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_bars, freq='1h')
        
        # Generate price data with trend and volatility
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, n_bars)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Generate volume
        volume = np.random.lognormal(10, 1, n_bars)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        })
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_multi_timeframe_data(self, timeframes: List[str]) -> dict:
        """
        Resample data to multiple timeframes
        
        Args:
            timeframes: List of timeframe strings (e.g., ['5m', '1h', '4h'])
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_ohlcv first.")
        
        mtf_data = {}
        for tf in timeframes:
            # Simple resampling (in production, use proper timeframe conversion)
            mtf_data[tf] = self.data.copy()
        
        return mtf_data
