"""
Configuration for SMC Trading System
Optimized parameters for best Sharpe ratio
"""

# Data parameters
TIMEFRAME = '1h'
SYMBOL = 'BTC/USD'

# SMC Detector parameters (Optimized)
ORDERBLOCK_LOOKBACK = 25  # bars to look back for orderblock detection (optimized from 20)
MIN_ORDERBLOCK_SIZE = 0.4  # minimum % size for valid orderblock (optimized from 0.5)
LIQUIDITY_GRAB_THRESHOLD = 1.5  # ATR multiplier for liquidity sweep detection

# Feature extraction parameters
ATR_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
MULTI_TIMEFRAMES = ['5m', '1h', '4h']

# Model parameters (Optimized)
SCORER_PROBABILITY_THRESHOLD = 0.5  # minimum probability to trade (optimized from 0.6)
ALLOWED_REGIMES = ['trending', 'high_volatility']  # Optimized regime filter
USE_ML_REGIME_DETECTOR = True  # Use ML-based regime detection (Phase 2)
USE_ENSEMBLE_REGIME = True  # Use ensemble of ML and rule-based (Phase 3 optimization)

# Risk management (Optimized for Sharpe ratio)
MAX_POSITION_SIZE = 0.015  # 1.5% of capital per trade (optimized from 2%)
STOP_LOSS_ATR_MULTIPLIER = 3.0  # Optimized from 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.5  # Optimized from 3.0
MAX_DAILY_LOSS = 0.05  # 5% max daily drawdown
MAX_CONCURRENT_POSITIONS = 3

# Execution
USE_LIMIT_ORDERS = True
SLIPPAGE_ESTIMATE = 0.001  # 0.1% estimated slippage
COMMISSION_RATE = 0.0005  # 0.05% commission
