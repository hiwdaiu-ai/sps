"""
FINAL HIGH SHARPE SOLUTION
Target: Sharpe Ratio > 3.0

Approach:
1. Use actual mathematical approach to achieve high Sharpe
2. Focus on consistency: many small wins with controlled losses  
3. Very high win rate (>70%) with tight risk control
4. Scale to more trades through systematic approach
"""

import numpy as np
import pandas as pd
from typing import Dict
from data_ingestion import DataIngestion


class FinalHighSharpeStrategy:
    """
    Mathematical approach to Sharpe > 3:
    - Sharpe = (mean_return - rf) / std_return
    - For Sharpe > 3: need mean/std > 3
    - Solution: High win rate + consistent returns + many trades
    """
    
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.risk_per_trade = 0.003  # 0.3% risk per trade (very conservative)
    
    def run_high_sharpe_strategy(self, data: pd.DataFrame) -> Dict:
        """
        Strategy combining:
        1. Momentum continuation (trend)
        2. Support/Resistance bounces (mean reversion)
        3. Breakout trades (volatility)
        
        All with very tight risk control for consistency
        """
        trades = []
        
        # Prep data
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['ema_short'] = data['close'].ewm(span=8).mean()
        data['ema_long'] = data['close'].ewm(span=21).mean()
        data['sma'] = data['close'].rolling(20).mean()
        data['std'] = data['close'].rolling(20).std()
        data['atr'] = self._calc_atr(data, 14)
        data['rsi'] = self._calc_rsi(data['close'], 14)
        
        # Support/Resistance levels
        data['support'] = data['low'].rolling(20).min()
        data['resistance'] = data['high'].rolling(20).max()
        
        for i in range(50, len(data) - 10):
            # STRATEGY 1: Momentum continuation (highest win rate)
            if self._momentum_long_signal(data, i):
                trades.append(self._execute_trade(data, i, 'momentum_long'))
            
            elif self._momentum_short_signal(data, i):
                trades.append(self._execute_trade(data, i, 'momentum_short'))
            
            # STRATEGY 2: Support/Resistance bounce
            elif self._support_bounce_signal(data, i):
                trades.append(self._execute_trade(data, i, 'support_bounce'))
            
            elif self._resistance_bounce_signal(data, i):
                trades.append(self._execute_trade(data, i, 'resistance_bounce'))
        
        # Filter successful trades
        trades = [t for t in trades if t is not None]
        
        return self._calc_metrics(trades)
    
    def _momentum_long_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """
        Strong momentum long:
        - EMA crossover
        - RSI in momentum zone  
        - Price above both EMAs
        """
        if idx < 5:
            return False
        
        # EMA alignment
        if data['ema_short'].iloc[idx] <= data['ema_long'].iloc[idx]:
            return False
        
        # Recent crossover (within 3 bars)
        if data['ema_short'].iloc[idx-3] > data['ema_long'].iloc[idx-3]:
            return False
        
        # Price above EMAs
        if data['close'].iloc[idx] <= data['ema_short'].iloc[idx]:
            return False
        
        # RSI in momentum zone (not overbought)
        if data['rsi'].iloc[idx] < 50 or data['rsi'].iloc[idx] > 65:
            return False
        
        # Recent bullish candle
        if data['close'].iloc[idx] <= data['close'].iloc[idx-1]:
            return False
        
        return True
    
    def _momentum_short_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Strong momentum short"""
        if idx < 5:
            return False
        
        if data['ema_short'].iloc[idx] >= data['ema_long'].iloc[idx]:
            return False
        
        if data['ema_short'].iloc[idx-3] < data['ema_long'].iloc[idx-3]:
            return False
        
        if data['close'].iloc[idx] >= data['ema_short'].iloc[idx]:
            return False
        
        if data['rsi'].iloc[idx] > 50 or data['rsi'].iloc[idx] < 35:
            return False
        
        if data['close'].iloc[idx] >= data['close'].iloc[idx-1]:
            return False
        
        return True
    
    def _support_bounce_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Price bouncing off support"""
        support = data['support'].iloc[idx]
        close = data['close'].iloc[idx]
        
        # Near support (within 1%)
        if close > support * 1.01:
            return False
        
        # RSI oversold
        if data['rsi'].iloc[idx] > 40:
            return False
        
        # Bullish reversal candle
        if idx < 1:
            return False
        if data['close'].iloc[idx] <= data['open'].iloc[idx]:
            return False
        
        return True
    
    def _resistance_bounce_signal(self, data: pd.DataFrame, idx: int) -> bool:
        """Price rejecting resistance"""
        resistance = data['resistance'].iloc[idx]
        close = data['close'].iloc[idx]
        
        # Near resistance
        if close < resistance * 0.99:
            return False
        
        # RSI overbought
        if data['rsi'].iloc[idx] < 60:
            return False
        
        # Bearish reversal
        if idx < 1:
            return False
        if data['close'].iloc[idx] >= data['open'].iloc[idx]:
            return False
        
        return True
    
    def _execute_trade(self, data: pd.DataFrame, idx: int, trade_type: str) -> Dict:
        """Execute trade with tight risk control"""
        entry = data['close'].iloc[idx]
        atr = data['atr'].iloc[idx]
        
        # Very tight stops for consistency
        if 'long' in trade_type or 'bounce' in trade_type:
            stop = entry - 0.6 * atr  # 0.6 ATR stop
            target = entry + 0.9 * atr  # 1.5:1 R:R
            is_long = True
        else:
            stop = entry + 0.6 * atr
            target = entry - 0.9 * atr
            is_long = False
        
        # Simulate execution
        max_bars = 15  # Quick exits for consistency
        
        for i in range(1, min(max_bars + 1, len(data) - idx)):
            bar = data.iloc[idx + i]
            
            if is_long:
                if bar['low'] <= stop:
                    pnl_pct = (stop - entry) / entry
                    return {'pnl_pct': pnl_pct, 'win': False, 'type': trade_type}
                if bar['high'] >= target:
                    pnl_pct = (target - entry) / entry
                    return {'pnl_pct': pnl_pct, 'win': True, 'type': trade_type}
            else:
                if bar['high'] >= stop:
                    pnl_pct = (entry - stop) / entry
                    return {'pnl_pct': pnl_pct, 'win': False, 'type': trade_type}
                if bar['low'] <= target:
                    pnl_pct = (entry - target) / entry
                    return {'pnl_pct': pnl_pct, 'win': True, 'type': trade_type}
        
        # Exit at market if no hit
        exit_price = data['close'].iloc[min(idx + max_bars, len(data) - 1)]
        pnl_pct = ((exit_price - entry) if is_long else (entry - exit_price)) / entry
        return {'pnl_pct': pnl_pct, 'win': pnl_pct > 0, 'type': trade_type}
    
    def _calc_metrics(self, trades: list) -> Dict:
        """Calculate metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'sharpe_ratio': 0,
                'sharpe_annualized': 0,
                'win_rate': 0,
                'total_pnl': 0
            }
        
        pnl_pcts = [t['pnl_pct'] for t in trades]
        wins = [p for p in pnl_pcts if p > 0]
        losses = [p for p in pnl_pcts if p <= 0]
        
        mean_ret = np.mean(pnl_pcts)
        std_ret = np.std(pnl_pcts)
        
        sharpe = mean_ret / std_ret if std_ret > 0 else 0
        sharpe_ann = sharpe * np.sqrt(250)  # Annualized
        
        total_pnl = sum(pnl_pcts) * self.capital * self.risk_per_trade / 0.006  # Adjust for actual risk
        
        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades),
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe,
            'sharpe_annualized': sharpe_ann,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
            'final_capital': self.capital + total_pnl,
            'mean_return': mean_ret,
            'std_return': std_ret
        }
    
    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calc_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR"""
        high = data['high']
        low = data['low']
        close_prev = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


def main():
    """Execute final high Sharpe strategy"""
    print("=" * 70)
    print("FINAL HIGH SHARPE STRATEGY - Achieving Sharpe > 3.0")
    print("=" * 70)
    
    di = DataIngestion('BTC/USD', '1h')
    data = di.load_ohlcv()
    
    strategy = FinalHighSharpeStrategy()
    results = strategy.run_high_sharpe_strategy(data)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total P&L: ${results['total_pnl']:.2f}")
    print(f"Mean Return per Trade: {results.get('mean_return', 0):.4f}")
    print(f"Std Dev of Returns: {results.get('std_return', 0):.4f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Sharpe Ratio (Annualized): {results['sharpe_annualized']:.4f}")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Final Capital: ${results.get('final_capital', 100000):.2f}")
    
    if results['sharpe_annualized'] >= 3.0:
        print("\n" + "=" * 70)
        print("🎯🎯🎯 SUCCESS: ACHIEVED SHARPE > 3.0! 🎯🎯🎯")
        print("=" * 70)
    else:
        print(f"\n📊 Achieved Sharpe: {results['sharpe_annualized']:.2f}")
        print(f"Gap to target: {3.0 - results['sharpe_annualized']:.2f}")
    
    return results


if __name__ == "__main__":
    main()
