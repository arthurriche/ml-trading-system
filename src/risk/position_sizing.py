"""
Position Sizing Module

This module provides various position sizing strategies including:
- Kelly Criterion
- Risk Parity
- Fixed Fractional
- Volatility Targeting
- Maximum Drawdown Control
- Portfolio Optimization-based sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class PositionSizer:
    """Base class for position sizing strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize position sizer.
        
        Args:
            initial_capital: Initial portfolio capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              **kwargs) -> float:
        """
        Calculate position size based on signal and market conditions.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            **kwargs: Additional parameters
            
        Returns:
            Position size in units
        """
        raise NotImplementedError("Subclasses must implement calculate_position_size")
    
    def update_capital(self, pnl: float):
        """Update current capital based on P&L."""
        self.current_capital += pnl


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing."""
    
    def __init__(self, initial_capital: float = 100000, max_kelly: float = 0.25):
        """
        Initialize Kelly Criterion sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            max_kelly: Maximum Kelly fraction (risk management)
        """
        super().__init__(initial_capital)
        self.max_kelly = max_kelly
        
    def calculate_kelly_fraction(self, 
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float) -> float:
        """
        Calculate Kelly fraction.
        
        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Kelly fraction
        """
        if avg_loss == 0:
            return 0
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply maximum Kelly constraint
        return min(kelly_fraction, self.max_kelly)
    
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              win_rate: float = 0.5,
                              avg_win: float = 0.02,
                              avg_loss: float = 0.01) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            win_rate: Historical win rate
            avg_win: Average win percentage
            avg_loss: Average loss percentage
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1:  # No significant signal
            return 0
        
        # Calculate Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # Adjust for signal strength
        adjusted_fraction = kelly_fraction * abs(signal)
        
        # Calculate position size
        position_value = self.current_capital * adjusted_fraction
        position_size = position_value / price
        
        return position_size * np.sign(signal)


class RiskParitySizer(PositionSizer):
    """Risk Parity position sizing."""
    
    def __init__(self, initial_capital: float = 100000, target_risk: float = 0.15):
        """
        Initialize Risk Parity sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            target_risk: Target annual portfolio risk
        """
        super().__init__(initial_capital)
        self.target_risk = target_risk
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              portfolio_volatility: float = None) -> float:
        """
        Calculate position size using Risk Parity.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            portfolio_volatility: Current portfolio volatility
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1 or volatility == 0:
            return 0
        
        # Calculate risk contribution
        if portfolio_volatility is None:
            portfolio_volatility = volatility
        
        # Risk parity allocation
        risk_contribution = self.target_risk / portfolio_volatility
        
        # Calculate position size
        position_value = self.current_capital * risk_contribution
        position_size = position_value / price
        
        return position_size * np.sign(signal)


class FixedFractionalSizer(PositionSizer):
    """Fixed Fractional position sizing."""
    
    def __init__(self, initial_capital: float = 100000, risk_per_trade: float = 0.02):
        """
        Initialize Fixed Fractional sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            risk_per_trade: Risk per trade as fraction of capital
        """
        super().__init__(initial_capital)
        self.risk_per_trade = risk_per_trade
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              stop_loss_pct: float = 0.05) -> float:
        """
        Calculate position size using Fixed Fractional method.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1:
            return 0
        
        # Calculate risk amount
        risk_amount = self.current_capital * self.risk_per_trade
        
        # Calculate position size based on stop loss
        if stop_loss_pct > 0:
            position_size = risk_amount / (price * stop_loss_pct)
        else:
            # Use volatility-based sizing
            position_size = risk_amount / (price * volatility)
        
        return position_size * np.sign(signal)


class VolatilityTargetingSizer(PositionSizer):
    """Volatility Targeting position sizing."""
    
    def __init__(self, initial_capital: float = 100000, target_volatility: float = 0.15):
        """
        Initialize Volatility Targeting sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            target_volatility: Target annual volatility
        """
        super().__init__(initial_capital)
        self.target_volatility = target_volatility
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              lookback_period: int = 252) -> float:
        """
        Calculate position size using Volatility Targeting.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility (annualized)
            lookback_period: Period for volatility calculation
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1 or volatility == 0:
            return 0
        
        # Calculate volatility scaling factor
        volatility_scaling = self.target_volatility / volatility
        
        # Calculate position size
        position_value = self.current_capital * volatility_scaling * abs(signal)
        position_size = position_value / price
        
        return position_size * np.sign(signal)


class MaximumDrawdownSizer(PositionSizer):
    """Maximum Drawdown Control position sizing."""
    
    def __init__(self, initial_capital: float = 100000, max_drawdown: float = 0.20):
        """
        Initialize Maximum Drawdown Control sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            max_drawdown: Maximum allowed drawdown
        """
        super().__init__(initial_capital)
        self.max_drawdown = max_drawdown
        self.peak_capital = initial_capital
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              current_drawdown: float = None) -> float:
        """
        Calculate position size with drawdown control.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1:
            return 0
        
        # Update peak capital
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Calculate current drawdown
        if current_drawdown is None:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Adjust position size based on drawdown
        if current_drawdown >= self.max_drawdown:
            return 0  # Stop trading if max drawdown reached
        
        # Reduce position size as drawdown approaches limit
        drawdown_factor = 1 - (current_drawdown / self.max_drawdown)
        
        # Calculate base position size
        base_position_value = self.current_capital * abs(signal) * 0.1  # 10% base allocation
        adjusted_position_value = base_position_value * drawdown_factor
        position_size = adjusted_position_value / price
        
        return position_size * np.sign(signal)


class PortfolioOptimizationSizer(PositionSizer):
    """Portfolio Optimization-based position sizing."""
    
    def __init__(self, initial_capital: float = 100000, target_return: float = 0.10):
        """
        Initialize Portfolio Optimization sizer.
        
        Args:
            initial_capital: Initial portfolio capital
            target_return: Target annual return
        """
        super().__init__(initial_capital)
        self.target_return = target_return
        self.asset_weights = {}
        
    def optimize_portfolio(self, 
                          returns_data: pd.DataFrame,
                          target_volatility: float = 0.15) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            returns_data: DataFrame of asset returns
            target_volatility: Target portfolio volatility
            
        Returns:
            Dictionary of optimal weights
        """
        n_assets = len(returns_data.columns)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        # Define objective function (minimize volatility)
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_vol
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.dot(expected_returns, x) - self.target_return}  # Target return
        ]
        
        # Bounds: weights between 0 and 1
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            return dict(zip(returns_data.columns, optimal_weights))
        else:
            # Fallback to equal weights
            return dict(zip(returns_data.columns, [1/n_assets] * n_assets))
    
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              asset_name: str = None,
                              optimal_weights: Dict[str, float] = None) -> float:
        """
        Calculate position size using portfolio optimization.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            asset_name: Name of the asset
            optimal_weights: Optimal portfolio weights
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1 or optimal_weights is None or asset_name is None:
            return 0
        
        # Get optimal weight for this asset
        if asset_name in optimal_weights:
            optimal_weight = optimal_weights[asset_name]
        else:
            optimal_weight = 0.1  # Default weight
        
        # Calculate position size
        position_value = self.current_capital * optimal_weight * abs(signal)
        position_size = position_value / price
        
        return position_size * np.sign(signal)


class AdaptivePositionSizer(PositionSizer):
    """Adaptive position sizing that combines multiple methods."""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize Adaptive position sizer.
        
        Args:
            initial_capital: Initial portfolio capital
        """
        super().__init__(initial_capital)
        self.kelly_sizer = KellyCriterionSizer(initial_capital)
        self.vol_sizer = VolatilityTargetingSizer(initial_capital)
        self.drawdown_sizer = MaximumDrawdownSizer(initial_capital)
        
    def calculate_position_size(self, 
                              signal: float,
                              price: float,
                              volatility: float,
                              market_regime: str = 'normal',
                              **kwargs) -> float:
        """
        Calculate position size using adaptive method.
        
        Args:
            signal: Trading signal (-1 to 1)
            price: Current asset price
            volatility: Asset volatility
            market_regime: Market regime ('normal', 'high_vol', 'crisis')
            **kwargs: Additional parameters
            
        Returns:
            Position size in units
        """
        if abs(signal) < 0.1:
            return 0
        
        # Calculate position sizes using different methods
        kelly_size = self.kelly_sizer.calculate_position_size(signal, price, volatility, **kwargs)
        vol_size = self.vol_sizer.calculate_position_size(signal, price, volatility, **kwargs)
        drawdown_size = self.drawdown_sizer.calculate_position_size(signal, price, volatility, **kwargs)
        
        # Weight based on market regime
        if market_regime == 'normal':
            weights = [0.4, 0.4, 0.2]  # Kelly, Vol, Drawdown
        elif market_regime == 'high_vol':
            weights = [0.2, 0.6, 0.2]  # More weight on volatility targeting
        else:  # crisis
            weights = [0.1, 0.3, 0.6]  # More weight on drawdown control
        
        # Calculate weighted average
        position_size = (weights[0] * kelly_size + 
                        weights[1] * vol_size + 
                        weights[2] * drawdown_size)
        
        return position_size


class PositionSizerFactory:
    """Factory class for creating position sizers."""
    
    @staticmethod
    def create_sizer(sizer_type: str, **kwargs) -> PositionSizer:
        """
        Create a position sizer based on type.
        
        Args:
            sizer_type: Type of sizer to create
            **kwargs: Sizer-specific parameters
            
        Returns:
            PositionSizer instance
        """
        sizers = {
            'kelly': KellyCriterionSizer,
            'risk_parity': RiskParitySizer,
            'fixed_fractional': FixedFractionalSizer,
            'volatility_targeting': VolatilityTargetingSizer,
            'max_drawdown': MaximumDrawdownSizer,
            'portfolio_optimization': PortfolioOptimizationSizer,
            'adaptive': AdaptivePositionSizer
        }
        
        if sizer_type not in sizers:
            raise ValueError(f"Unknown sizer type: {sizer_type}")
        
        return sizers[sizer_type](**kwargs)


# Utility functions
def calculate_optimal_leverage(returns: pd.Series, target_volatility: float = 0.15) -> float:
    """Calculate optimal leverage for volatility targeting."""
    current_volatility = returns.std() * np.sqrt(252)
    return target_volatility / current_volatility if current_volatility > 0 else 1.0


def calculate_risk_budget(volatilities: List[float], target_risk: float = 0.15) -> List[float]:
    """Calculate risk budget allocation."""
    total_vol = sum(volatilities)
    if total_vol == 0:
        return [1.0 / len(volatilities)] * len(volatilities)
    
    risk_budget = [vol / total_vol * target_risk for vol in volatilities]
    return risk_budget


# Example usage
def run_position_sizing_example():
    """Run example position sizing calculations."""
    # Sample data
    signal = 0.8
    price = 100.0
    volatility = 0.25
    current_capital = 100000
    
    # Test different sizers
    sizers = {
        'Kelly': KellyCriterionSizer(current_capital),
        'Risk Parity': RiskParitySizer(current_capital),
        'Fixed Fractional': FixedFractionalSizer(current_capital),
        'Volatility Targeting': VolatilityTargetingSizer(current_capital),
        'Max Drawdown': MaximumDrawdownSizer(current_capital),
        'Adaptive': AdaptivePositionSizer(current_capital)
    }
    
    results = {}
    for name, sizer in sizers.items():
        try:
            position_size = sizer.calculate_position_size(signal, price, volatility)
            position_value = abs(position_size * price)
            risk_exposure = position_value / current_capital
            
            results[name] = {
                'position_size': position_size,
                'position_value': position_value,
                'risk_exposure': risk_exposure
            }
            
            print(f"{name}:")
            print(f"  Position Size: {position_size:.2f} units")
            print(f"  Position Value: ${position_value:,.2f}")
            print(f"  Risk Exposure: {risk_exposure:.2%}")
            print()
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return results


if __name__ == "__main__":
    # Run example
    results = run_position_sizing_example()
