"""
Market Data Loader for ML Trading System

This module provides comprehensive market data loading and processing capabilities:
- Historical data fetching from multiple sources
- Real-time data streaming
- Data cleaning and preprocessing
- Technical indicator calculation
- Data validation and quality checks
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class MarketDataLoader:
    """
    Comprehensive market data loader for trading system.
    
    Supports multiple data sources:
    - Yahoo Finance (yfinance)
    - Alpha Vantage API
    - IEX Cloud API
    - Custom CSV files
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize market data loader.
        
        Args:
            api_keys: Dictionary of API keys for different data sources
        """
        self.api_keys = api_keys or {}
        self.cache = {}
        self.session = requests.Session()
        
    def load_data(self, 
                  symbol: str, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None,
                  interval: str = '1d',
                  source: str = 'yahoo') -> pd.DataFrame:
        """
        Load market data for a given symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '5m', etc.)
            source: Data source ('yahoo', 'alpha_vantage', 'iex', 'csv')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if source == 'yahoo':
                return self._load_yahoo_data(symbol, start_date, end_date, interval)
            elif source == 'alpha_vantage':
                return self._load_alpha_vantage_data(symbol, start_date, end_date, interval)
            elif source == 'iex':
                return self._load_iex_data(symbol, start_date, end_date, interval)
            elif source == 'csv':
                return self._load_csv_data(symbol, start_date, end_date)
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_yahoo_data(self, 
                        symbol: str, 
                        start_date: Optional[str], 
                        end_date: Optional[str],
                        interval: str) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_alpha_vantage_data(self, 
                                symbol: str, 
                                start_date: Optional[str], 
                                end_date: Optional[str],
                                interval: str) -> pd.DataFrame:
        """Load data from Alpha Vantage API."""
        if 'alpha_vantage' not in self.api_keys:
            logger.error("Alpha Vantage API key not provided")
            return pd.DataFrame()
        
        try:
            api_key = self.api_keys['alpha_vantage']
            
            # Determine function based on interval
            if interval == '1d':
                function = 'TIME_SERIES_DAILY'
            elif interval == '1h':
                function = 'TIME_SERIES_INTRADAY'
            else:
                function = 'TIME_SERIES_DAILY'
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            # Extract time series data
            time_series_key = [key for key in data.keys() if 'Time Series' in key][0]
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            
            # Rename columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_iex_data(self, 
                      symbol: str, 
                      start_date: Optional[str], 
                      end_date: Optional[str],
                      interval: str) -> pd.DataFrame:
        """Load data from IEX Cloud API."""
        if 'iex' not in self.api_keys:
            logger.error("IEX Cloud API key not provided")
            return pd.DataFrame()
        
        try:
            api_key = self.api_keys['iex']
            
            # Calculate date range
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/1y"
            params = {
                'token': api_key
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            if isinstance(data, dict) and 'error' in data:
                logger.error(f"IEX error: {data['error']}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Filter date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading IEX data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _load_csv_data(self, 
                      file_path: str, 
                      start_date: Optional[str], 
                      end_date: Optional[str]) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Filter date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {e}")
            return pd.DataFrame()
    
    def load_multiple_symbols(self, 
                            symbols: List[str], 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            interval: str = '1d',
                            source: str = 'yahoo') -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            source: Data source
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            logger.info(f"Loading data for {symbol}")
            data = self.load_data(symbol, start_date, end_date, interval, source)
            if not data.empty:
                data_dict[symbol] = data
            else:
                logger.warning(f"Failed to load data for {symbol}")
        
        return data_dict
    
    def get_latest_price(self, symbol: str, source: str = 'yahoo') -> float:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            source: Data source
            
        Returns:
            Latest closing price
        """
        try:
            data = self.load_data(symbol, source=source)
            if not data.empty:
                return data['close'].iloc[-1]
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with return calculations
        """
        if data.empty:
            return pd.DataFrame()
        
        # Calculate different types of returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['high_low_returns'] = (data['high'] - data['low']) / data['close'].shift(1)
        data['open_close_returns'] = (data['close'] - data['open']) / data['open']
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        if data.empty:
            return pd.DataFrame()
        
        # Moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and completeness.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'missing_values': {},
            'outliers': {},
            'data_quality_score': 0.0,
            'issues': []
        }
        
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Data is empty")
            return validation_results
        
        # Check for missing values
        missing_values = data.isnull().sum()
        validation_results['missing_values'] = missing_values.to_dict()
        
        if missing_values.sum() > 0:
            validation_results['issues'].append(f"Found {missing_values.sum()} missing values")
        
        # Check for outliers in price data
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                validation_results['outliers'][col] = len(outliers)
        
        # Calculate data quality score
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        quality_score = (total_cells - missing_cells) / total_cells
        validation_results['data_quality_score'] = quality_score
        
        # Check for price anomalies
        if 'close' in data.columns:
            price_changes = data['close'].pct_change().abs()
            large_changes = price_changes > 0.1  # 10% daily change
            if large_changes.sum() > 0:
                validation_results['issues'].append(f"Found {large_changes.sum()} large price changes (>10%)")
        
        return validation_results
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data
        
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Sort by index
        data = data.sort_index()
        
        # Handle missing values
        # Forward fill for OHLC data
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Fill volume with 0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Ensure price data is positive
        for col in ohlc_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        return data
    
    def resample_data(self, 
                     data: pd.DataFrame, 
                     frequency: str = '1D',
                     agg_method: str = 'ohlc') -> pd.DataFrame:
        """
        Resample data to different frequency.
        
        Args:
            data: Input DataFrame
            frequency: Target frequency ('1D', '1H', '5T', etc.)
            agg_method: Aggregation method ('ohlc', 'last', 'mean')
            
        Returns:
            Resampled DataFrame
        """
        if data.empty:
            return data
        
        if agg_method == 'ohlc':
            # Resample OHLCV data
            resampled = data.resample(frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        elif agg_method == 'last':
            resampled = data.resample(frequency).last()
        elif agg_method == 'mean':
            resampled = data.resample(frequency).mean()
        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")
        
        return resampled.dropna()


# Example usage
def load_sample_data():
    """Load sample data for testing."""
    loader = MarketDataLoader()
    
    # Load data for a few major stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    data_dict = loader.load_multiple_symbols(
        symbols, 
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    return data_dict


if __name__ == "__main__":
    # Example usage
    loader = MarketDataLoader()
    
    # Load data for Apple
    aapl_data = loader.load_data('AAPL', start_date='2023-01-01', end_date='2023-12-31')
    
    if not aapl_data.empty:
        print(f"Loaded {len(aapl_data)} rows of data for AAPL")
        print(f"Date range: {aapl_data.index[0]} to {aapl_data.index[-1]}")
        print(f"Latest close: ${aapl_data['close'].iloc[-1]:.2f}")
        
        # Add technical indicators
        aapl_data = loader.add_technical_indicators(aapl_data)
        print(f"Added technical indicators. Columns: {list(aapl_data.columns)}")
        
        # Validate data
        validation = loader.validate_data(aapl_data)
        print(f"Data quality score: {validation['data_quality_score']:.2f}")
    else:
        print("Failed to load data")
