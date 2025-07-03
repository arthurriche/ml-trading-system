"""
Performance metrics for ML Trading System.

This module provides comprehensive performance evaluation tools including:
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis
- Risk metrics (VaR, CVaR, volatility)
- Trading performance metrics (win rate, profit factor, etc.)
- Portfolio attribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """Comprehensive performance metrics calculator for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, 
                             returns: pd.Series,
                             benchmark_returns: Optional[pd.Series] = None,
                             trading_costs: float = 0.001) -> Dict:
        """
        Calculate all performance metrics for a strategy.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            trading_costs: Transaction costs per trade
            
        Returns:
            Dictionary containing all performance metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(returns))
        
        # Trading metrics
        metrics.update(self._calculate_trading_metrics(returns, trading_costs))
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate basic return metrics."""
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        avg_return = returns.mean()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_daily_return': avg_return,
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk metrics."""
        volatility = returns.std() * np.sqrt(252)
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_drawdown': max_drawdown,
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted return metrics."""
        excess_returns = returns - self.risk_free_rate / 252
        
        # Sharpe ratio
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
        else:
            sortino_ratio = np.inf
        
        # Calmar ratio
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = np.inf
        
        # Information ratio (if benchmark provided)
        information_ratio = None
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict:
        """Calculate drawdown-related metrics."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        avg_drawdown = drawdown[drawdown < 0].mean()
        
        # Drawdown duration
        drawdown_periods = (drawdown < 0).sum()
        avg_drawdown_duration = drawdown_periods / len(returns)
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration': drawdown_periods,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_trading_metrics(self, returns: pd.Series, trading_costs: float) -> Dict:
        """Calculate trading-specific metrics."""
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Expected value
        expected_value = returns.mean()
        
        # Risk of ruin (simplified)
        risk_of_ruin = self._calculate_risk_of_ruin(returns)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expected_value': expected_value,
            'risk_of_ruin': risk_of_ruin,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades
        }
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict:
        """Calculate benchmark comparison metrics."""
        # Alpha and Beta
        excess_returns = returns - self.risk_free_rate / 252
        excess_benchmark = benchmark_returns - self.risk_free_rate / 252
        
        # Calculate beta
        covariance = np.cov(excess_returns, excess_benchmark)[0, 1]
        benchmark_variance = np.var(excess_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha
        alpha = excess_returns.mean() - beta * excess_benchmark.mean()
        
        # Information ratio
        tracking_error = (excess_returns - excess_benchmark).std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() - excess_benchmark.mean()) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
        
        # Correlation
        correlation = returns.corr(benchmark_returns)
        
        # R-squared
        r_squared = correlation ** 2
        
        return {
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'correlation': correlation,
            'r_squared': r_squared
        }
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        recovery_times = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                recovery_times.append(i - drawdown_start)
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_risk_of_ruin(self, returns: pd.Series, 
                               initial_capital: float = 1.0,
                               ruin_threshold: float = 0.5) -> float:
        """Calculate simplified risk of ruin."""
        # Monte Carlo simulation for risk of ruin
        n_simulations = 10000
        ruin_count = 0
        
        for _ in range(n_simulations):
            capital = initial_capital
            for ret in returns:
                capital *= (1 + ret)
                if capital <= ruin_threshold:
                    ruin_count += 1
                    break
        
        return ruin_count / n_simulations


class PortfolioAttribution:
    """Portfolio attribution analysis for multi-asset portfolios."""
    
    def __init__(self):
        """Initialize portfolio attribution analyzer."""
        pass
    
    def calculate_attribution(self, 
                            portfolio_weights: Dict[str, float],
                            asset_returns: pd.DataFrame,
                            benchmark_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate portfolio attribution analysis.
        
        Args:
            portfolio_weights: Dictionary of asset weights in portfolio
            asset_returns: DataFrame of asset returns
            benchmark_weights: Dictionary of benchmark weights
            
        Returns:
            Dictionary containing attribution analysis
        """
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(portfolio_weights, asset_returns)
        
        # Calculate benchmark returns if provided
        benchmark_returns = None
        if benchmark_weights:
            benchmark_returns = self._calculate_portfolio_returns(benchmark_weights, asset_returns)
        
        # Calculate attribution
        attribution = self._calculate_attribution_breakdown(
            portfolio_weights, asset_returns, benchmark_weights
        )
        
        return {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'attribution': attribution
        }
    
    def _calculate_portfolio_returns(self, weights: Dict[str, float], 
                                   returns: pd.DataFrame) -> pd.Series:
        """Calculate weighted portfolio returns."""
        portfolio_returns = pd.Series(0.0, index=returns.index)
        
        for asset, weight in weights.items():
            if asset in returns.columns:
                portfolio_returns += weight * returns[asset]
        
        return portfolio_returns
    
    def _calculate_attribution_breakdown(self, 
                                       portfolio_weights: Dict[str, float],
                                       asset_returns: pd.DataFrame,
                                       benchmark_weights: Optional[Dict[str, float]]) -> Dict:
        """Calculate attribution breakdown."""
        attribution = {}
        
        # Allocation effect
        if benchmark_weights:
            allocation_effect = {}
            for asset in portfolio_weights:
                if asset in benchmark_weights:
                    weight_diff = portfolio_weights[asset] - benchmark_weights[asset]
                    asset_return = asset_returns[asset].mean() * 252  # Annualized
                    allocation_effect[asset] = weight_diff * asset_return
            
            attribution['allocation_effect'] = allocation_effect
        
        # Selection effect
        selection_effect = {}
        for asset, weight in portfolio_weights.items():
            if asset in asset_returns.columns:
                asset_return = asset_returns[asset].mean() * 252
                selection_effect[asset] = weight * asset_return
        
        attribution['selection_effect'] = selection_effect
        
        return attribution


class RiskMetrics:
    """Advanced risk metrics calculator."""
    
    def __init__(self):
        """Initialize risk metrics calculator."""
        pass
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_volatility(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling volatility."""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series, 
                      window: int = 252) -> pd.Series:
        """Calculate rolling beta."""
        covariance = returns.rolling(window=window).cov(market_returns)
        market_variance = market_returns.rolling(window=window).var()
        return covariance / market_variance
    
    def calculate_correlation(self, returns: pd.Series, benchmark_returns: pd.Series,
                            window: int = 252) -> pd.Series:
        """Calculate rolling correlation."""
        return returns.rolling(window=window).corr(benchmark_returns)


# Utility functions
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    return excess_returns.mean() / downside_deviation * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate."""
    return (returns > 0).mean()


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else np.inf


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 500), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 500), index=dates)
    
    # Calculate performance metrics
    metrics_calc = PerformanceMetrics()
    metrics = metrics_calc.calculate_all_metrics(returns, benchmark_returns)
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}") 