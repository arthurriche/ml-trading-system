"""
Visualization utilities for the ML Trading System.

This module provides comprehensive visualization tools for:
- Performance metrics and backtesting results
- Technical indicators and price charts
- Model predictions and trading signals
- Risk analysis and portfolio composition
- Market data analysis and feature importance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


class TradingVisualizer:
    """Main visualization class for trading system analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
        
    def plot_price_chart(self, 
                        data: pd.DataFrame,
                        title: str = "Price Chart with Technical Indicators",
                        indicators: Optional[List[str]] = None,
                        signals: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create an interactive price chart with technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            title: Chart title
            indicators: List of technical indicators to plot
            signals: DataFrame with buy/sell signals
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(title, 'Volume', 'Technical Indicators'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Technical indicators
        if indicators:
            for indicator in indicators:
                if indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[indicator],
                            name=indicator,
                            line=dict(width=1)
                        ),
                        row=3, col=1
                    )
        
        # Trading signals
        if signals is not None:
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]
            
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['price'],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color=self.colors['success']
                        )
                    ),
                    row=1, col=1
                )
            
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['price'],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color=self.colors['danger']
                        )
                    ),
                    row=1, col=1
                )
        
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_performance_metrics(self, 
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                title: str = "Performance Analysis") -> go.Figure:
        """
        Create comprehensive performance analysis charts.
        
        Args:
            returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            title: Chart title
            
        Returns:
            Plotly figure with multiple subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative Returns',
                'Rolling Sharpe Ratio',
                'Drawdown Analysis',
                'Return Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name='Portfolio',
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            fig.add_trace(
                go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'])
                ),
                row=1, col=1
            )
        
        # Rolling Sharpe ratio
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name='Rolling Sharpe',
                line=dict(color=self.colors['info'])
            ),
            row=1, col=2
        )
        
        # Drawdown analysis
        drawdown = self._calculate_drawdown(cumulative_returns)
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name='Drawdown',
                fill='tonexty',
                line=dict(color=self.colors['danger'])
            ),
            row=2, col=1
        )
        
        # Return distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name='Returns',
                nbinsx=50,
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def plot_model_predictions(self,
                              actual: np.ndarray,
                              predicted: np.ndarray,
                              dates: Optional[pd.DatetimeIndex] = None,
                              title: str = "Model Predictions vs Actual") -> go.Figure:
        """
        Plot model predictions against actual values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Date index for x-axis
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(actual), freq='D')
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['primary'])
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predicted,
                mode='lines',
                name='Predicted',
                line=dict(color=self.colors['secondary'])
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self,
                               feature_names: List[str],
                               importance_scores: np.ndarray,
                               title: str = "Feature Importance",
                               top_n: int = 20) -> go.Figure:
        """
        Plot feature importance for ML models.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            title: Chart title
            top_n: Number of top features to display
            
        Returns:
            Plotly figure
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:top_n]]
        top_scores = importance_scores[sorted_idx[:top_n]]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=top_scores,
                y=top_features,
                orientation='h',
                marker_color=self.colors['primary']
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(top_features) * 20)
        )
        
        return fig
    
    def plot_correlation_matrix(self,
                               data: pd.DataFrame,
                               title: str = "Feature Correlation Matrix") -> go.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with features
            title: Chart title
            
        Returns:
            Plotly figure
        """
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600
        )
        
        return fig
    
    def plot_risk_metrics(self,
                         portfolio_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         title: str = "Risk Analysis") -> go.Figure:
        """
        Create comprehensive risk analysis charts.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Value at Risk (VaR)',
                'Expected Shortfall (CVaR)',
                'Rolling Volatility',
                'Beta Analysis'
            )
        )
        
        # VaR analysis
        var_levels = [0.01, 0.05, 0.1]
        var_values = [np.percentile(portfolio_returns, level * 100) for level in var_levels]
        
        fig.add_trace(
            go.Bar(
                x=[f'{level*100}%' for level in var_levels],
                y=var_values,
                name='VaR',
                marker_color=self.colors['danger']
            ),
            row=1, col=1
        )
        
        # Rolling volatility
        rolling_vol = portfolio_returns.rolling(window=252).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                name='Rolling Volatility',
                line=dict(color=self.colors['warning'])
            ),
            row=2, col=1
        )
        
        # Beta analysis (if benchmark provided)
        if benchmark_returns is not None:
            rolling_beta = self._calculate_rolling_beta(portfolio_returns, benchmark_returns)
            fig.add_trace(
                go.Scatter(
                    x=rolling_beta.index,
                    y=rolling_beta.values,
                    name='Rolling Beta',
                    line=dict(color=self.colors['info'])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        return (rolling_mean / rolling_std) * np.sqrt(252)
    
    def _calculate_drawdown(self, cumulative_returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown
    
    def _calculate_rolling_beta(self, portfolio_returns: pd.Series, 
                               benchmark_returns: pd.Series, 
                               window: int = 252) -> pd.Series:
        """Calculate rolling beta."""
        covariance = portfolio_returns.rolling(window=window).cov(benchmark_returns)
        benchmark_variance = benchmark_returns.rolling(window=window).var()
        return covariance / benchmark_variance


def create_dashboard(portfolio_data: Dict,
                    model_results: Dict,
                    risk_metrics: Dict) -> go.Figure:
    """
    Create a comprehensive trading dashboard.
    
    Args:
        portfolio_data: Dictionary containing portfolio performance data
        model_results: Dictionary containing model predictions and metrics
        risk_metrics: Dictionary containing risk analysis results
        
    Returns:
        Plotly figure with dashboard layout
    """
    visualizer = TradingVisualizer()
    
    # Create subplots for dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Portfolio Performance',
            'Model Predictions',
            'Risk Metrics',
            'Feature Importance',
            'Correlation Matrix',
            'Trading Signals'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add portfolio performance
    if 'returns' in portfolio_data:
        cumulative_returns = (1 + portfolio_data['returns']).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                name='Portfolio',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    # Add model predictions
    if 'actual' in model_results and 'predicted' in model_results:
        fig.add_trace(
            go.Scatter(
                x=range(len(model_results['actual'])),
                y=model_results['actual'],
                name='Actual',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=range(len(model_results['predicted'])),
                y=model_results['predicted'],
                name='Predicted',
                line=dict(color='red')
            ),
            row=1, col=2
        )
    
    # Add risk metrics
    if 'var' in risk_metrics:
        fig.add_trace(
            go.Bar(
                x=['95% VaR', '99% VaR'],
                y=[risk_metrics['var']['95'], risk_metrics['var']['99']],
                name='VaR',
                marker_color='red'
            ),
            row=2, col=1
        )
    
    # Add feature importance
    if 'feature_importance' in model_results:
        importance = model_results['feature_importance']
        top_features = list(importance.keys())[:10]
        top_scores = list(importance.values())[:10]
        
        fig.add_trace(
            go.Bar(
                x=top_scores,
                y=top_features,
                orientation='h',
                name='Feature Importance',
                marker_color='green'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="ML Trading System Dashboard",
        height=1200,
        showlegend=True
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, format: str = 'html'):
    """
    Save plot to file.
    
    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format ('html', 'png', 'pdf')
    """
    if format == 'html':
        fig.write_html(filename)
    elif format == 'png':
        fig.write_image(filename)
    elif format == 'pdf':
        fig.write_image(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage functions
def plot_sample_data():
    """Create sample visualizations for demonstration."""
    visualizer = TradingVisualizer()
    
    # Generate sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = 100 + np.cumsum(np.random.randn(500) * 0.02)
    returns = pd.Series(np.random.randn(500) * 0.01, index=dates)
    
    # Create sample OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(500) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(500) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(500) * 0.02)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    # Create sample signals
    signals = pd.DataFrame({
        'signal': np.random.choice([-1, 0, 1], 500, p=[0.1, 0.8, 0.1]),
        'price': prices
    }, index=dates)
    
    # Generate plots
    price_chart = visualizer.plot_price_chart(data, signals=signals)
    performance_chart = visualizer.plot_performance_metrics(returns)
    
    return price_chart, performance_chart


if __name__ == "__main__":
    # Example usage
    price_chart, performance_chart = plot_sample_data()
    price_chart.show()
    performance_chart.show() 