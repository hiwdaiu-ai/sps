"""
Configuration for SMC Trading System
"""

# Data parameters
TIMEFRAME = '1h'
SYMBOL = 'BTC/USD'

# SMC Detector parameters
ORDERBLOCK_LOOKBACK = 20  # bars to look back for orderblock detection
MIN_ORDERBLOCK_SIZE = 0.5  # minimum % size for valid orderblock
LIQUIDITY_GRAB_THRESHOLD = 1.5  # ATR multiplier for liquidity sweep detection

# Feature extraction parameters
ATR_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
MULTI_TIMEFRAMES = ['5m', '1h', '4h']

# Model parameters
SCORER_PROBABILITY_THRESHOLD = 0.6  # minimum probability to trade
ALLOWED_REGIMES = ['trending', 'high_volatility']
USE_ML_REGIME_DETECTOR = True  # Use ML-based regime detection (Phase 2)

# Risk management
MAX_POSITION_SIZE = 0.02  # 2% of capital per trade
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
MAX_DAILY_LOSS = 0.05  # 5% max daily drawdown
MAX_CONCURRENT_POSITIONS = 3

# Execution
USE_LIMIT_ORDERS = True
SLIPPAGE_ESTIMATE = 0.001  # 0.1% estimated slippage
COMMISSION_RATE = 0.0005  # 0.05% commission
