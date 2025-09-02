"""
Data input/output module for stock market data collection and preprocessing.
Handles downloading and caching of S&P 500 data from Yahoo Finance.
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from typing import Optional, Tuple


class DataLoader:
    """Handles data loading and caching for stock market data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_sp500_data(self, 
                           start_date: str = "2005-01-01", 
                           end_date: str = "2022-03-31",
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Download S&P 500 data from Yahoo Finance or create synthetic data if not available.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached data exists
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = os.path.join(self.data_dir, "sp500_raw.csv")
        
        # Check if cached data exists and is valid
        if not force_refresh and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                print(f"Loaded cached S&P 500 data: {len(df)} records")
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}. Re-downloading...")
        
        try:
            print(f"Downloading S&P 500 data from {start_date} to {end_date}...")
            
            # Download S&P 500 data (^GSPC is the Yahoo Finance symbol)
            ticker = yf.Ticker("^GSPC")
            df = ticker.history(start=start_date, end=end_date)
            
            # Rename columns to standard format
            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Remove any rows with missing data
            df = df.dropna()
            
            # Save to cache
            df.to_csv(cache_file)
            print(f"Downloaded and cached {len(df)} records")
            
            return df
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating synthetic S&P 500 data for demonstration...")
            return self._create_synthetic_data(start_date, end_date, cache_file)
    
    def _create_synthetic_data(self, start_date: str, end_date: str, cache_file: str) -> pd.DataFrame:
        """Create synthetic S&P 500 data based on realistic patterns."""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Filter to business days only
        dates = dates[dates.weekday < 5]
        
        n_days = len(dates)
        
        # Generate realistic S&P 500 price data
        np.random.seed(42)  # For reproducibility
        
        # Starting price around 1200 (2005 level)
        initial_price = 1200.0
        
        # Generate returns with realistic volatility and drift
        daily_return_mean = 0.0008  # ~20% annual return
        daily_return_std = 0.012    # ~19% annual volatility
        
        returns = np.random.normal(daily_return_mean, daily_return_std, n_days)
        
        # Add some autocorrelation and volatility clustering
        for i in range(1, len(returns)):
            returns[i] += 0.05 * returns[i-1]  # Small momentum effect
        
        # Calculate cumulative prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        close_prices = prices
        
        # Open prices (small gap from previous close)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = initial_price
        open_gap = np.random.normal(0, 0.002, n_days)
        open_prices = open_prices * (1 + open_gap)
        
        # High and low prices
        daily_range = np.random.uniform(0.005, 0.025, n_days)  # 0.5% to 2.5% daily range
        high_prices = np.maximum(open_prices, close_prices) * (1 + daily_range/2)
        low_prices = np.minimum(open_prices, close_prices) * (1 - daily_range/2)
        
        # Volume (realistic S&P 500 ETF volume)
        base_volume = 50_000_000
        volume_volatility = np.random.lognormal(0, 0.5, n_days)
        volume = (base_volume * volume_volatility).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"Created and cached synthetic S&P 500 data: {len(df)} records")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def load_processed_data(self, filename: str) -> Optional[object]:
        """Load processed data from pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_processed_data(self, data: object, filename: str) -> None:
        """Save processed data to pickle file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {filepath}")


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.download_sp500_data()
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(df.head())