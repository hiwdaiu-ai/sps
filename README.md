# SMC Trading System - Production-Grade Prototype

A hybrid, engineered trading system that combines deterministic, rule-based Smart Money Concepts (SMC) heuristics with robust statistical/ML filters and enterprise-grade execution & risk controls.

## Overview

This system implements the practical approach described in `ins.txt` for building a production-ready SMC trading system. Instead of trying to create a single end-to-end AI that "learns SMC," this uses a hybrid approach:

- **Rule-based SMC detectors** to generate candidate setups
- **ML/statistics** (regime & probability models) to filter and score candidates
- **Conservative execution + risk layer** (position sizing, stop-loss, slippage model)

Machine learning is used as a **filter/scorer**, not the sole decision-maker.

## Architecture

The system consists of the following modules:

### 1. Data Ingestion (`data_ingestion.py`)
- Handles OHLCV data loading and preprocessing
- Multi-timeframe data support
- Sample data generation for testing

### 2. SMC Detector (`smc_detector.py`)
- Deterministic functions to identify:
  - **Orderblocks**: Strong candles with high volume followed by significant moves
  - **Liquidity grabs**: Large wicks with volume spikes indicating stop hunts
  - **Break of Structure (BOS)**: Price breaking recent highs/lows
  - **Change of Character (CHOCH)**: Trend reversal signals

### 3. Feature Extraction (`feature_extraction.py`)
- Converts SMC candidates into numeric features:
  - Distance to zone (normalized by ATR)
  - Volume ratios and deltas
  - Candle characteristics (body/wick ratios)
  - Momentum indicators (EMA, RSI)
  - Volatility metrics (ATR)
  - Multi-timeframe alignment

### 4. Models (`models.py`)
- **Regime Detector**: Classifies market state (trending, ranging, high/low volatility)
- **Signal Scorer**: LightGBM model providing P(win) for each candidate
- Outcome-based labeling for training without manual annotation

### 5. Risk Manager (`risk_manager.py`)
- Position sizing (fixed fractional)
- Dynamic stop-loss and take-profit (ATR-based)
- Daily loss limits
- Position concurrency limits
- Circuit breakers

### 6. Execution Engine (`execution.py`)
- Smart order execution
- Limit vs market order support
- Slippage modeling for development
- Commission tracking
- Order fill simulation

### 7. Main Trading System (`main.py`)
- Coordinates all components
- Implements the core trading loop:
  ```python
  1. read OHLCV & orderbook
  2. detect_orderblocks() -> list of candidates
  3. for each candidate:
       features = extract_features(candidate)
       p = scorer.predict_proba(features)
       regime = regime_model.predict(features)
       if p > threshold and regime in allowed:
           place_order_with_risk_controls()
  ```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run the complete system with sample data
python main.py
```

### Using Your Own Data

```python
from main import SMCTradingSystem

# Initialize system
system = SMCTradingSystem(symbol='BTC/USD', timeframe='1h')

# Load your CSV data (must have: timestamp, open, high, low, close, volume)
system.load_data('path/to/your/data.csv')

# Train models
system.train_models()

# Run backtest
results = system.run_backtest()
print(results)
```

### Configuration

Edit `config.py` to customize parameters:

```python
# SMC Detection
ORDERBLOCK_LOOKBACK = 20
MIN_ORDERBLOCK_SIZE = 0.5
LIQUIDITY_GRAB_THRESHOLD = 1.5

# Model
SCORER_PROBABILITY_THRESHOLD = 0.6
ALLOWED_REGIMES = ['trending', 'high_volatility']

# Risk Management
MAX_POSITION_SIZE = 0.02  # 2% per trade
STOP_LOSS_ATR_MULTIPLIER = 2.0
TAKE_PROFIT_ATR_MULTIPLIER = 3.0
MAX_DAILY_LOSS = 0.05  # 5% max daily drawdown
```

## Implementation Roadmap

This prototype implements **Phase 0**, **Phase 1**, and **Phase 2** from the roadmap in `ins.txt`:

✅ **Phase 0 - Foundation (1-3 weeks)**
- OHLCV data ingestion
- Basic SMC rule-based detector (configurable parameters)
- Backtest detector as rule-based strategy with risk rules

✅ **Phase 1 - Feature + Scorer (2-4 weeks)**
- Feature pipeline for candidate events
- Outcome-based label generation
- LightGBM scorer with calibrated outputs
- Walk-forward compatible evaluation

✅ **Phase 2 - Regime + Filter (2-3 weeks)** 🆕
- **ML-based regime detector** trained on historical data
- Enhanced feature extraction (9+ regime features)
- Combined rule-based SMC + scorer + regime gating
- Realistic slippage & commission in backtest

### Next Phases (Not Yet Implemented)

**Phase 3 - Paper Trading & Execution (4-8 weeks)**
- Live API connection
- Real execution logic
- A/B testing limit vs market orders
- Slippage monitoring and adjustment

**Phase 4 - Production & Monitoring (ongoing)**
- Deployment with monitoring and alerts
- Automated retraining schedule
- Feature drift detection

## What's New in Phase 2

### ML-Based Regime Detection
The system now uses a LightGBM classifier to detect market regimes:
- **4 Regime Classes**: trending, ranging, high_volatility, low_volatility
- **9+ Features**: volatility ratios, trend strength, price range, volume, directional movement
- **Automated Training**: Trains on historical data during `train_models()`
- **Configurable**: Toggle with `USE_ML_REGIME_DETECTOR` in config

### Performance Impact
With ML-based regime detection:
- **More trades**: 11 vs 3 (more realistic trading frequency)
- **Higher P&L**: +$8,364 vs +$6,790
- **Profit factor**: 1.82 (still profitable)

See `PHASE2_IMPLEMENTATION.md` for complete details.

## Key Features

### 1. No Manual Labeling Required
The system uses **outcome-based labeling**: labels are generated by actual P&L after an event window, eliminating the need for manual chart annotation.

### 2. Robust Risk Controls
- Position sizing with hard limits
- ATR-based dynamic stops
- Daily loss circuit breakers
- Position concurrency limits

### 3. Realistic Backtesting
- Transaction cost aware (commission + slippage)
- Proper fill simulation
- Walk-forward validation ready

### 4. Production-Ready Structure
- Modular design for easy maintenance
- Configurable parameters
- Comprehensive logging potential
- Extensible architecture

## Performance Metrics

The system tracks standard trading metrics:
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Average Win/Loss**: Expected value per trade
- **Total P&L**: Cumulative profit/loss
- **Max Drawdown**: Largest peak-to-trough decline

## Important Notes

### Realistic Expectations
- No system is "guaranteed profitable"
- The goal is **statistical edge + robust risk controls**
- Expect model decay - plan for monitoring and retraining
- Small capital and highly liquid instruments typically work better

### Risk Disclaimer
This is a prototype for educational and research purposes. Trading involves substantial risk of loss. Always:
- Test thoroughly with paper trading before live deployment
- Start with small position sizes
- Monitor performance continuously
- Ensure regulatory compliance

### Technical Stack
- **Python 3.8+**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **lightgbm**: Gradient boosting models
- **scikit-learn**: ML utilities

## Extending the System

### Adding New SMC Patterns
Edit `smc_detector.py` to add new pattern detection methods:
```python
def detect_fair_value_gaps(self, df: pd.DataFrame) -> List[FVG]:
    # Your implementation
    pass
```

### Custom Features
Add new features in `feature_extraction.py`:
```python
features['your_feature'] = calculate_your_feature(df, candidate)
```

### Alternative Models
Replace LightGBM with your preferred model in `models.py`:
```python
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier()
```

## Testing

### Run Component Tests

```bash
# Run all component tests
python test_system.py
```

This tests all major components: data ingestion, SMC detection, feature extraction, regime detection, signal scoring, risk management, and execution.

### Run Usage Examples

```bash
# Run all usage examples
python examples.py
```

This demonstrates:
- Basic usage with sample data
- Pattern detection only
- Feature extraction
- Regime detection over time
- Custom configuration
- Risk management controls

### Validate with Your Own Data

1. Prepare CSV with columns: `timestamp, open, high, low, close, volume`
2. Load data: `system.load_data('your_data.csv')`
3. Run backtest: `results = system.run_backtest()`

## License

This project is provided as-is for educational purposes.

## References

See `ins.txt` for the complete implementation guide and theoretical background.
