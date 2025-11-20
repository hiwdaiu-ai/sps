"""
Example usage of the SMC Trading System
Demonstrates different use cases
"""

from main import SMCTradingSystem
import pandas as pd
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic usage with sample data"""
    print("=" * 60)
    print("Example 1: Basic Usage with Sample Data")
    print("=" * 60)
    
    # Initialize system
    system = SMCTradingSystem(symbol='BTC/USD', timeframe='1h')
    
    # Load sample data (auto-generated)
    system.load_data()
    
    # Train models
    system.train_models()
    
    # Run backtest
    results = system.run_backtest()
    
    # Print results
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")


def example_2_detect_patterns():
    """Example 2: Detect SMC patterns without trading"""
    print("\n" + "=" * 60)
    print("Example 2: Pattern Detection Only")
    print("=" * 60)
    
    system = SMCTradingSystem()
    data = system.load_data()
    
    # Detect orderblocks
    orderblocks = system.smc_detector.detect_orderblocks(data)
    print(f"\nDetected {len(orderblocks)} orderblocks:")
    for i, ob in enumerate(orderblocks[:5]):  # Show first 5
        direction = "Bullish" if ob.is_bullish else "Bearish"
        print(f"  {i+1}. {direction} @ {ob.timestamp}, Strength: {ob.strength:.2f}")
    
    # Detect liquidity grabs
    liquidity_grabs = system.smc_detector.detect_liquidity_grabs(data)
    print(f"\nDetected {len(liquidity_grabs)} liquidity grabs:")
    for i, lg in enumerate(liquidity_grabs[:5]):  # Show first 5
        print(f"  {i+1}. {lg.direction.capitalize()} @ {lg.timestamp}, Volume Spike: {lg.volume_spike:.2f}x")
    
    # Detect BOS/CHOCH
    bos_events = system.smc_detector.detect_bos_choch(data)
    print(f"\nDetected {len(bos_events)} BOS/CHOCH events:")
    for i, event in enumerate(bos_events[:5]):  # Show first 5
        print(f"  {i+1}. {event['type']} {event['direction']} @ {event['timestamp']}")


def example_3_feature_extraction():
    """Example 3: Extract features from a single orderblock"""
    print("\n" + "=" * 60)
    print("Example 3: Feature Extraction")
    print("=" * 60)
    
    system = SMCTradingSystem()
    data = system.load_data()
    
    # Get first orderblock
    orderblocks = system.smc_detector.detect_orderblocks(data)
    if orderblocks:
        ob = orderblocks[0]
        ob_idx = data.index.get_loc(ob.timestamp)
        
        # Extract features
        if ob_idx + 10 < len(data):
            features = system.feature_extractor.extract_features(
                ob, data, ob_idx + 5
            )
            
            print(f"\nFeatures for orderblock at {ob.timestamp}:")
            for key, value in features.items():
                print(f"  {key}: {value:.4f}")


def example_4_regime_detection():
    """Example 4: Detect market regime over time"""
    print("\n" + "=" * 60)
    print("Example 4: Regime Detection")
    print("=" * 60)
    
    system = SMCTradingSystem()
    data = system.load_data()
    
    # Sample every 50 bars
    print("\nMarket regime over time:")
    for i in range(50, len(data), 50):
        regime = system.regime_detector.detect_regime(data, i)
        timestamp = data.index[i]
        print(f"  {timestamp}: {regime}")


def example_5_custom_parameters():
    """Example 5: Using custom configuration"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Modify config
    import config
    original_threshold = config.SCORER_PROBABILITY_THRESHOLD
    config.SCORER_PROBABILITY_THRESHOLD = 0.7  # More conservative
    
    print(f"Using probability threshold: {config.SCORER_PROBABILITY_THRESHOLD}")
    
    system = SMCTradingSystem()
    system.load_data()
    system.train_models()
    results = system.run_backtest()
    
    print(f"\nWith conservative threshold (0.7):")
    print(f"  Total trades: {results['total_trades']}")
    print(f"  Win rate: {results['win_rate']:.2%}")
    print(f"  Total P&L: ${results['total_pnl']:.2f}")
    
    # Restore original
    config.SCORER_PROBABILITY_THRESHOLD = original_threshold


def example_6_risk_controls():
    """Example 6: Testing risk management controls"""
    print("\n" + "=" * 60)
    print("Example 6: Risk Management")
    print("=" * 60)
    
    from risk_manager import RiskManager
    
    risk_mgr = RiskManager(
        max_position_size=0.02,  # 2% per trade
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        max_daily_loss=0.05,  # 5% max daily loss
        max_concurrent=3
    )
    
    capital = 100000
    entry_price = 50000
    atr = 1000
    
    # Calculate stops
    stops = risk_mgr.calculate_stops(entry_price, atr, is_long=True)
    print(f"\nFor a long position at ${entry_price}:")
    print(f"  Stop Loss: ${stops['stop_loss']:.2f}")
    print(f"  Take Profit: ${stops['take_profit']:.2f}")
    print(f"  Risk per trade: ${(entry_price - stops['stop_loss']):.2f}")
    
    # Calculate position size
    position_size = risk_mgr.calculate_position_size(
        capital, entry_price, stops['stop_loss']
    )
    print(f"  Position size: {position_size:.4f} units")
    print(f"  Position value: ${position_size * entry_price:.2f}")
    
    # Test daily loss limit
    print(f"\nRisk Controls:")
    print(f"  Can trade: {risk_mgr.can_trade(capital)}")
    
    # Simulate large loss
    risk_mgr.update_pnl(-capital * 0.06)  # 6% loss
    print(f"  After 6% daily loss, can trade: {risk_mgr.can_trade(capital)}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("SMC Trading System - Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_1_basic_usage()
    example_2_detect_patterns()
    example_3_feature_extraction()
    example_4_regime_detection()
    example_5_custom_parameters()
    example_6_risk_controls()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
