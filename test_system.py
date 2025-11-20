"""
Basic tests for SMC Trading System components
Run with: python test_system.py
"""

import sys
import pandas as pd
import numpy as np
from data_ingestion import DataIngestion
from smc_detector import SMCDetector
from feature_extraction import FeatureExtractor
from models import RegimeDetector, SignalScorer
from risk_manager import RiskManager
from execution import ExecutionEngine, Order


def test_data_ingestion():
    """Test data ingestion module"""
    print("Testing Data Ingestion...")
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    assert len(data) == 1000, "Should generate 1000 bars"
    assert 'open' in data.columns, "Should have 'open' column"
    assert 'high' in data.columns, "Should have 'high' column"
    assert 'low' in data.columns, "Should have 'low' column"
    assert 'close' in data.columns, "Should have 'close' column"
    assert 'volume' in data.columns, "Should have 'volume' column"
    
    print("  ✓ Data ingestion works correctly")


def test_smc_detector():
    """Test SMC detector"""
    print("Testing SMC Detector...")
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    detector = SMCDetector(lookback=20, min_size=0.5)
    
    # Test orderblock detection
    orderblocks = detector.detect_orderblocks(data)
    assert isinstance(orderblocks, list), "Should return a list"
    assert len(orderblocks) > 0, "Should detect some orderblocks"
    
    # Test liquidity grab detection
    grabs = detector.detect_liquidity_grabs(data)
    assert isinstance(grabs, list), "Should return a list"
    
    # Test BOS/CHOCH detection
    bos_events = detector.detect_bos_choch(data)
    assert isinstance(bos_events, list), "Should return a list"
    assert len(bos_events) > 0, "Should detect some BOS events"
    
    print(f"  ✓ Detected {len(orderblocks)} orderblocks")
    print(f"  ✓ Detected {len(grabs)} liquidity grabs")
    print(f"  ✓ Detected {len(bos_events)} BOS/CHOCH events")


def test_feature_extraction():
    """Test feature extraction"""
    print("Testing Feature Extraction...")
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    detector = SMCDetector()
    orderblocks = detector.detect_orderblocks(data)
    
    extractor = FeatureExtractor()
    
    if orderblocks:
        ob = orderblocks[0]
        ob_idx = data.index.get_loc(ob.timestamp)
        
        if ob_idx + 10 < len(data):
            features = extractor.extract_features(ob, data, ob_idx + 5)
            
            assert isinstance(features, dict), "Should return a dictionary"
            assert 'distance_to_zone' in features, "Should have distance feature"
            assert 'atr' in features, "Should have ATR feature"
            assert 'rsi' in features, "Should have RSI feature"
            
            print(f"  ✓ Extracted {len(features)} features")


def test_regime_detector():
    """Test regime detector"""
    print("Testing Regime Detector...")
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    # Test rule-based detector
    detector = RegimeDetector(use_ml=False)
    regime = detector.detect_regime(data)
    
    assert regime in ['trending', 'ranging', 'high_volatility', 'low_volatility'], \
        "Should return valid regime"
    
    print(f"  ✓ Rule-based regime: {regime}")
    
    # Test ML-based detector
    ml_detector = RegimeDetector(use_ml=True)
    ml_detector.train(data, lookback=100)
    ml_regime = ml_detector.detect_regime(data)
    
    assert ml_regime in ['trending', 'ranging', 'high_volatility', 'low_volatility'], \
        "Should return valid regime"
    
    print(f"  ✓ ML-based regime: {ml_regime}")
    print(f"  ✓ Regime detector trained successfully")


def test_signal_scorer():
    """Test signal scorer"""
    print("Testing Signal Scorer...")
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    detector = SMCDetector()
    orderblocks = detector.detect_orderblocks(data)
    
    scorer = SignalScorer()
    
    # Generate labels
    labels = scorer.generate_training_labels(data, orderblocks, horizon=10)
    assert len(labels) > 0, "Should generate some labels"
    
    print(f"  ✓ Generated {len(labels)} training labels")
    print(f"  ✓ Positive samples: {labels.sum()}, Negative samples: {len(labels) - labels.sum()}")


def test_risk_manager():
    """Test risk manager"""
    print("Testing Risk Manager...")
    
    risk_mgr = RiskManager(
        max_position_size=0.02,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0
    )
    
    capital = 100000
    entry_price = 50000
    atr = 1000
    
    # Test position sizing
    stops = risk_mgr.calculate_stops(entry_price, atr, is_long=True)
    position_size = risk_mgr.calculate_position_size(capital, entry_price, stops['stop_loss'])
    
    assert position_size > 0, "Position size should be positive"
    assert stops['stop_loss'] < entry_price, "Stop loss should be below entry for long"
    assert stops['take_profit'] > entry_price, "Take profit should be above entry for long"
    
    # Test trading limits
    assert risk_mgr.can_trade(capital) == True, "Should allow trading initially"
    
    # Simulate large loss
    risk_mgr.update_pnl(-capital * 0.06)
    assert risk_mgr.can_trade(capital) == False, "Should stop trading after daily loss limit"
    
    print("  ✓ Position sizing works correctly")
    print("  ✓ Risk limits enforced properly")


def test_execution_engine():
    """Test execution engine"""
    print("Testing Execution Engine...")
    
    engine = ExecutionEngine(use_limit_orders=True, slippage_estimate=0.001)
    
    order = Order(
        symbol='BTC/USD',
        side='buy',
        order_type='limit',
        quantity=0.1,
        price=50000,
        stop_loss=48000,
        take_profit=53000
    )
    
    fill = engine.execute_order(order, market_price=50000)
    
    assert fill is not None, "Should return a fill"
    assert fill.fill_quantity == order.quantity, "Fill quantity should match order"
    assert fill.commission > 0, "Should calculate commission"
    
    print("  ✓ Order execution works correctly")
    print(f"  ✓ Fill price: ${fill.fill_price:.2f}, Commission: ${fill.commission:.2f}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SMC Trading System - Component Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_data_ingestion,
        test_smc_detector,
        test_feature_extraction,
        test_regime_detector,
        test_signal_scorer,
        test_risk_manager,
        test_execution_engine
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Test error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
