"""
Momentum Trading Strategy

This module implements various momentum-based trading strategies including:
- Price momentum strategies
- Relative strength index (RSI) strategies
- Moving average crossover strategies
- MACD-based strategies
- Volume-weighted momentum strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod


class MomentumStrategy(ABC):
    """Abstract base class for momentum trading strategies."""
    
    def __init__(self, lookback_period: int = 20, threshold: float = 0.02):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            threshold: Signal threshold for trading decisions
        """
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.positions = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on momentum indicators."""
        pass
    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calculate strategy returns based on signals."""
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns
        return strategy_returns
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest for the strategy."""
        signals = self.generate_signals(data)
        returns = self.calculate_returns(data, signals)
        
        # Calculate performance metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'signals': signals,
            'returns': returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()


class PriceMomentumStrategy(MomentumStrategy):
    """Simple price momentum strategy based on price changes."""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on price momentum."""
        # Calculate momentum as price change over lookback period
        momentum = data['close'].pct_change(self.lookback_period)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[momentum > self.threshold] = 1  # Buy signal
        signals[momentum < -self.threshold] = -1  # Sell signal
        
        return signals


class RSIMomentumStrategy(MomentumStrategy):
    """RSI-based momentum strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize RSI momentum strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            oversold: Oversold threshold
            overbought: Overbought threshold
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI indicator."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI."""
        rsi = self.calculate_rsi(data)
        
        signals = pd.Series(0, index=data.index)
        signals[rsi < self.oversold] = 1  # Buy when oversold
        signals[rsi > self.overbought] = -1  # Sell when overbought
        
        return signals


class MovingAverageCrossoverStrategy(MomentumStrategy):
    """Moving average crossover momentum strategy."""
    
    def __init__(self, short_period: int = 10, long_period: int = 50):
        """
        Initialize moving average crossover strategy.
        
        Args:
            short_period: Short-term moving average period
            long_period: Long-term moving average period
        """
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover."""
        short_ma = data['close'].rolling(window=self.short_period).mean()
        long_ma = data['close'].rolling(window=self.long_period).mean()
        
        signals = pd.Series(0, index=data.index)
        signals[short_ma > long_ma] = 1  # Buy when short MA > long MA
        signals[short_ma < long_ma] = -1  # Sell when short MA < long MA
        
        return signals


class MACDMomentumStrategy(MomentumStrategy):
    """MACD-based momentum strategy."""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD momentum strategy.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        fast_ema = data['close'].ewm(span=self.fast_period).mean()
        slow_ema = data['close'].ewm(span=self.slow_period).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD."""
        macd_line, signal_line, histogram = self.calculate_macd(data)
        
        signals = pd.Series(0, index=data.index)
        signals[(macd_line > signal_line) & (histogram > 0)] = 1  # Buy signal
        signals[(macd_line < signal_line) & (histogram < 0)] = -1  # Sell signal
        
        return signals


class VolumeWeightedMomentumStrategy(MomentumStrategy):
    """Volume-weighted momentum strategy."""
    
    def __init__(self, lookback_period: int = 20, volume_threshold: float = 1.5):
        """
        Initialize volume-weighted momentum strategy.
        
        Args:
            lookback_period: Period for momentum calculation
            volume_threshold: Volume threshold multiplier
        """
        super().__init__(lookback_period)
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on volume-weighted momentum."""
        # Calculate volume-weighted momentum
        volume_ma = data['volume'].rolling(window=self.lookback_period).mean()
        high_volume = data['volume'] > (volume_ma * self.volume_threshold)
        
        # Calculate price momentum
        momentum = data['close'].pct_change(self.lookback_period)
        
        # Generate signals only on high volume days
        signals = pd.Series(0, index=data.index)
        signals[(momentum > self.threshold) & high_volume] = 1
        signals[(momentum < -self.threshold) & high_volume] = -1
        
        return signals


class MultiTimeframeMomentumStrategy(MomentumStrategy):
    """Multi-timeframe momentum strategy."""
    
    def __init__(self, timeframes: List[int] = [5, 20, 50]):
        """
        Initialize multi-timeframe momentum strategy.
        
        Args:
            timeframes: List of lookback periods for different timeframes
        """
        super().__init__()
        self.timeframes = timeframes
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on multi-timeframe momentum."""
        signals = pd.Series(0, index=data.index)
        
        # Calculate momentum for each timeframe
        momentum_signals = []
        for period in self.timeframes:
            momentum = data['close'].pct_change(period)
            momentum_signal = pd.Series(0, index=data.index)
            momentum_signal[momentum > self.threshold] = 1
            momentum_signal[momentum < -self.threshold] = -1
            momentum_signals.append(momentum_signal)
        
        # Combine signals (majority vote)
        for i in range(len(data)):
            votes = [signal.iloc[i] for signal in momentum_signals]
            positive_votes = sum(1 for vote in votes if vote > 0)
            negative_votes = sum(1 for vote in votes if vote < 0)
            
            if positive_votes > negative_votes:
                signals.iloc[i] = 1
            elif negative_votes > positive_votes:
                signals.iloc[i] = -1
        
        return signals


class AdaptiveMomentumStrategy(MomentumStrategy):
    """Adaptive momentum strategy that adjusts parameters based on market conditions."""
    
    def __init__(self, volatility_lookback: int = 50):
        """
        Initialize adaptive momentum strategy.
        
        Args:
            volatility_lookback: Period for volatility calculation
        """
        super().__init__()
        self.volatility_lookback = volatility_lookback
    
    def calculate_adaptive_threshold(self, data: pd.DataFrame) -> pd.Series:
        """Calculate adaptive threshold based on market volatility."""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_lookback).std()
        
        # Adjust threshold based on volatility
        base_threshold = 0.02
        adaptive_threshold = base_threshold * (volatility / volatility.mean())
        
        return adaptive_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals with adaptive threshold."""
        adaptive_threshold = self.calculate_adaptive_threshold(data)
        momentum = data['close'].pct_change(self.lookback_period)
        
        signals = pd.Series(0, index=data.index)
        
        for i in range(len(data)):
            if momentum.iloc[i] > adaptive_threshold.iloc[i]:
                signals.iloc[i] = 1
            elif momentum.iloc[i] < -adaptive_threshold.iloc[i]:
                signals.iloc[i] = -1
        
        return signals


class MomentumStrategyFactory:
    """Factory class for creating momentum strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, **kwargs) -> MomentumStrategy:
        """
        Create a momentum strategy based on type.
        
        Args:
            strategy_type: Type of strategy to create
            **kwargs: Strategy-specific parameters
            
        Returns:
            MomentumStrategy instance
        """
        strategies = {
            'price': PriceMomentumStrategy,
            'rsi': RSIMomentumStrategy,
            'ma_crossover': MovingAverageCrossoverStrategy,
            'macd': MACDMomentumStrategy,
            'volume_weighted': VolumeWeightedMomentumStrategy,
            'multi_timeframe': MultiTimeframeMomentumStrategy,
            'adaptive': AdaptiveMomentumStrategy
        }
        
        if strategy_type not in strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategies[strategy_type](**kwargs)


# Example usage and testing
def run_momentum_strategy_example():
    """Run example momentum strategy backtest."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Create sample price data with trend
    trend = np.linspace(100, 150, 500)
    noise = np.random.normal(0, 2, 500)
    prices = trend + noise
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, 500)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, 500))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, 500))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # Test different strategies
    strategies = {
        'Price Momentum': PriceMomentumStrategy(lookback_period=20),
        'RSI': RSIMomentumStrategy(),
        'MA Crossover': MovingAverageCrossoverStrategy(),
        'MACD': MACDMomentumStrategy(),
        'Volume Weighted': VolumeWeightedMomentumStrategy(),
        'Multi Timeframe': MultiTimeframeMomentumStrategy(),
        'Adaptive': AdaptiveMomentumStrategy()
    }
    
    results = {}
    for name, strategy in strategies.items():
        try:
            result = strategy.backtest(data)
            results[name] = result
            print(f"{name}:")
            print(f"  Total Return: {result['total_return']:.4f}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.4f}")
            print()
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return results


if __name__ == "__main__":
    # Run example
    results = run_momentum_strategy_example()
