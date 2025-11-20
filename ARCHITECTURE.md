# SMC Trading System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SMC Trading System                        │
│                     (main.py - Orchestrator)                     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │     Data     │  │     SMC      │  │   Feature    │
      │  Ingestion   │  │   Detector   │  │  Extraction  │
      │──────────────│  │──────────────│  │──────────────│
      │ • Load OHLCV │  │ • Orderblocks│  │ • Distance   │
      │ • Multi-TF   │  │ • Liq Grabs  │  │ • Volume     │
      │ • Sample Gen │  │ • BOS/CHOCH  │  │ • Momentum   │
      └──────────────┘  └──────────────┘  └──────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │   Regime     │  │    Signal    │  │     Risk     │
      │   Detector   │  │    Scorer    │  │   Manager    │
      │──────────────│  │──────────────│  │──────────────│
      │ • Trending   │  │ • LightGBM   │  │ • Position   │
      │ • Ranging    │  │ • Prob Score │  │   Sizing     │
      │ • Volatility │  │ • Calibrated │  │ • Stops/TP   │
      └──────────────┘  └──────────────┘  └──────────────┘
                                  │
                                  ▼
                        ┌──────────────┐
                        │  Execution   │
                        │    Engine    │
                        │──────────────│
                        │ • Orders     │
                        │ • Slippage   │
                        │ • Commission │
                        └──────────────┘
```

## Data Flow

```
1. OHLCV Data → Data Ingestion
                      ↓
2. Price Patterns → SMC Detector → [Candidates]
                      ↓
3. Candidates → Feature Extraction → [Feature Matrix]
                      ↓
4. Features → Regime Detector → [Market State]
              Signal Scorer → [Probability]
                      ↓
5. If (Prob > Threshold) AND (Regime in Allowed):
                      ↓
6. Risk Manager → Calculate Position Size & Stops
                      ↓
7. Execution Engine → Place Order
                      ↓
8. Track P&L & Update Risk Metrics
```

## Module Responsibilities

### Data Ingestion (`data_ingestion.py`)
- Load historical OHLCV data
- Generate sample data for testing
- Multi-timeframe support

### SMC Detector (`smc_detector.py`)
**Input:** OHLCV DataFrame  
**Output:** List of SMC pattern candidates  
**Patterns Detected:**
- Orderblocks (bullish/bearish)
- Liquidity grabs/sweeps
- Break of Structure (BOS)
- Change of Character (CHOCH)

### Feature Extraction (`feature_extraction.py`)
**Input:** SMC candidate + OHLCV data  
**Output:** Feature dictionary (16+ features)  
**Features:**
- Distance to zone (ATR-normalized)
- Volume ratios
- Candle characteristics
- Momentum (EMA, RSI)
- Volatility (ATR)

### Regime Detector (`models.py`)
**Input:** OHLCV data  
**Output:** Market regime classification  
**Regimes:**
- Trending
- Ranging
- High volatility
- Low volatility

### Signal Scorer (`models.py`)
**Input:** Feature matrix  
**Output:** Probability of success (0-1)  
**Model:** LightGBM with calibrated probabilities  
**Training:** Outcome-based labels (no manual annotation)

### Risk Manager (`risk_manager.py`)
**Input:** Capital, entry price, ATR  
**Output:** Position size, stop-loss, take-profit  
**Controls:**
- Max position size (% of capital)
- Daily loss limits
- Position concurrency limits
- ATR-based dynamic stops

### Execution Engine (`execution.py`)
**Input:** Order details, market price  
**Output:** Fill with slippage & commission  
**Features:**
- Limit/market orders
- Slippage simulation
- Commission calculation
- Fill tracking

## Configuration (`config.py`)

All parameters are centralized:
- SMC detection thresholds
- Feature extraction periods
- Model parameters
- Risk management limits
- Execution settings

## Testing Structure

```
test_system.py
├─ test_data_ingestion()
├─ test_smc_detector()
├─ test_feature_extraction()
├─ test_regime_detector()
├─ test_signal_scorer()
├─ test_risk_manager()
└─ test_execution_engine()
```

## Example Usage

```
examples.py
├─ Example 1: Basic usage with sample data
├─ Example 2: Pattern detection only
├─ Example 3: Feature extraction
├─ Example 4: Regime detection
├─ Example 5: Custom parameters
└─ Example 6: Risk controls
```

## Design Principles

1. **Hybrid Approach**: Rule-based detection + ML filtering
2. **Modularity**: Independent, testable components
3. **No Manual Labeling**: Outcome-based training
4. **Risk First**: Conservative execution with multiple safeguards
5. **Transparency**: Interpretable decisions at each step
6. **Extensibility**: Easy to add new patterns, features, or models
