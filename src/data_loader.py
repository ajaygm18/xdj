"""
Real data loader using yfinance for stock market data.
Implements proper data fetching, preprocessing, and validation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class YFinanceDataLoader:
    """
    Yahoo Finance data loader for real stock market data.
    
    Provides clean, validated OHLCV data for stock market prediction models.
    """
    
    def __init__(self, symbol: str = "^GSPC", period: str = "10y"):
        """
        Initialize data loader.
        
        Args:
            symbol: Yahoo Finance symbol (default: ^GSPC for S&P 500)
            period: Data period (default: 10y for 10 years)
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.info = {}
    
    def download_data(self, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading {self.symbol} data from Yahoo Finance...")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(self.symbol)
            
            # Download data
            if start_date and end_date:
                self.data = ticker.history(start=start_date, end=end_date)
                print(f"Downloaded data from {start_date} to {end_date}")
            else:
                self.data = ticker.history(period=self.period)
                print(f"Downloaded data for period: {self.period}")
            
            # Get ticker info
            try:
                self.info = ticker.info
            except:
                self.info = {"symbol": self.symbol}
            
            # Validate data
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Clean column names (Yahoo Finance returns capitalized names)
            self.data.columns = [col.lower() for col in self.data.columns]
            
            # Remove any rows with NaN values
            initial_len = len(self.data)
            self.data = self.data.dropna()
            final_len = len(self.data)
            
            if initial_len > final_len:
                print(f"Removed {initial_len - final_len} rows with missing data")
            
            print(f"Successfully downloaded {len(self.data)} trading days")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"Columns: {list(self.data.columns)}")
            
            return self.data
            
        except Exception as e:
            print(f"Error downloading data for {self.symbol}: {str(e)}")
            raise
    
    def get_ohlcv_data(self) -> pd.DataFrame:
        """
        Get clean OHLCV data suitable for technical analysis.
        
        Returns:
            DataFrame with open, high, low, close, volume columns
        """
        if self.data is None:
            raise ValueError("No data available. Call download_data() first.")
        
        # Select OHLCV columns
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in ohlcv_columns if col in self.data.columns]
        
        if len(available_columns) < 4:  # Need at least OHLC
            raise ValueError(f"Insufficient OHLCV data. Available: {available_columns}")
        
        ohlcv_data = self.data[available_columns].copy()
        
        # Validate data quality
        self._validate_ohlcv_data(ohlcv_data)
        
        return ohlcv_data
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> None:
        """
        Validate OHLCV data for common issues.
        
        Args:
            data: OHLCV DataFrame to validate
        """
        print("Validating OHLCV data...")
        
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    print(f"Warning: {negative_count} non-positive values in {col}")
        
        # Check for impossible OHLC relationships
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= max(open, close)
            invalid_high = ((data['high'] < data['open']) | 
                           (data['high'] < data['close'])).sum()
            if invalid_high > 0:
                print(f"Warning: {invalid_high} rows where high < max(open, close)")
            
            # Low should be <= min(open, close)
            invalid_low = ((data['low'] > data['open']) | 
                          (data['low'] > data['close'])).sum()
            if invalid_low > 0:
                print(f"Warning: {invalid_low} rows where low > min(open, close)")
        
        # Check for extreme outliers (>5 standard deviations)
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                mean_val = data[col].mean()
                std_val = data[col].std()
                outliers = np.abs(data[col] - mean_val) > 5 * std_val
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    print(f"Warning: {outlier_count} potential outliers in {col}")
        
        print("Data validation completed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the downloaded data.
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            return {}
        
        stats = {
            'symbol': self.symbol,
            'total_days': len(self.data),
            'date_range': {
                'start': self.data.index[0].date().isoformat(),
                'end': self.data.index[-1].date().isoformat()
            },
            'price_statistics': {},
            'volume_statistics': {}
        }
        
        # Price statistics
        if 'close' in self.data.columns:
            close_prices = self.data['close']
            stats['price_statistics'] = {
                'min_price': float(close_prices.min()),
                'max_price': float(close_prices.max()),
                'mean_price': float(close_prices.mean()),
                'std_price': float(close_prices.std()),
                'total_return': float((close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100)
            }
        
        # Volume statistics
        if 'volume' in self.data.columns:
            volume = self.data['volume']
            stats['volume_statistics'] = {
                'min_volume': int(volume.min()),
                'max_volume': int(volume.max()),
                'mean_volume': int(volume.mean()),
                'std_volume': int(volume.std())
            }
        
        return stats
    
    def save_data(self, filepath: str) -> None:
        """
        Save downloaded data to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Call download_data() first.")
        
        self.data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Data loaded from {filepath}")
            print(f"Loaded {len(self.data)} rows")
            return self.data
        except Exception as e:
            print(f"Error loading data from {filepath}: {str(e)}")
            raise


def get_sp500_data(years: int = 10, save_to_file: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to get S&P 500 data.
    
    Args:
        years: Number of years of historical data
        save_to_file: Whether to save data to CSV file
        
    Returns:
        Tuple of (OHLCV DataFrame, statistics dict)
    """
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Load data
    loader = YFinanceDataLoader("^GSPC")
    data = loader.download_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    # Get OHLCV data
    ohlcv_data = loader.get_ohlcv_data()
    
    # Get statistics
    stats = loader.get_statistics()
    
    # Save to file if requested
    if save_to_file:
        filename = f"sp500_data_{years}y_{datetime.now().strftime('%Y%m%d')}.csv"
        loader.save_data(f"data/{filename}")
    
    return ohlcv_data, stats


def get_multiple_stocks(symbols: list, period: str = "5y") -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple stock symbols.
    
    Args:
        symbols: List of Yahoo Finance symbols
        period: Data period
        
    Returns:
        Dictionary mapping symbols to OHLCV DataFrames
    """
    results = {}
    
    for symbol in symbols:
        try:
            print(f"\nDownloading {symbol}...")
            loader = YFinanceDataLoader(symbol, period)
            loader.download_data()
            results[symbol] = loader.get_ohlcv_data()
            print(f"✓ {symbol}: {len(results[symbol])} days")
        except Exception as e:
            print(f"✗ {symbol}: Failed - {str(e)}")
            results[symbol] = None
    
    return results


if __name__ == "__main__":
    # Test data loading
    print("Testing Yahoo Finance data loader...")
    
    try:
        # Test S&P 500 data
        ohlcv_data, stats = get_sp500_data(years=5, save_to_file=False)
        
        print("\nData sample:")
        print(ohlcv_data.head())
        print(f"\nData shape: {ohlcv_data.shape}")
        
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\nData loader test completed successfully!")
        
    except Exception as e:
        print(f"Data loader test failed: {str(e)}")
        import traceback
        traceback.print_exc()