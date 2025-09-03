"""
USA Market Data Loading (S&P 500 only) - Paper Compliant
Focuses exclusively on S&P 500 market as requested.
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class USAMarketDataLoader:
    """Loads data from USA market (S&P 500) only - focused implementation."""
    
    # USA market only as requested
    USA_MARKET = {
        'SP500': '^GSPC',      # S&P 500 (U.S.) - Primary focus
    }
    
    def __init__(self):
        """Initialize USA market data loader."""
        pass
    
    def download_market_data(self, market_code: str = "SP500", start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download S&P 500 data (USA market only).
        
        Args:
            market_code: Market code (only 'SP500' supported)
            start_date: Start date in 'YYYY-MM-DD' format (default: paper start)
            end_date: End date in 'YYYY-MM-DD' format (default: paper end)
            
        Returns:
            DataFrame with OHLCV data
        """
        if market_code not in self.USA_MARKET:
            print(f"Warning: Only SP500 (USA) market supported. Using SP500 instead of {market_code}")
            market_code = "SP500"
        
        symbol = self.USA_MARKET[market_code]
        print(f"Downloading USA market data ({symbol}) from {start_date} to {end_date}...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {market_code} ({symbol})")
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have the required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns for {market_code}")
            
            print(f"✓ Downloaded {len(data)} samples for USA market (SP500)")
            return data[required_cols]
            
        except Exception as e:
            print(f"❌ Failed to download USA market data: {e}")
            raise
    
    def download_usa_market(self, start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download USA market data (S&P 500) with paper-compliant timeframe.
        
        Args:
            start_date: Start date (paper default: 2005-01-01)
            end_date: End date (paper default: 2022-03-31)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading USA market (S&P 500) data from {start_date} to {end_date}...")
        return self.download_market_data("SP500", start_date, end_date)
    
    def get_market_info(self) -> Dict[str, Dict]:
        """Get information about USA market only."""
        return {
            'SP500': {
                'name': 'S&P 500',
                'country': 'United States',
                'symbol': '^GSPC',
                'description': 'Standard & Poor\'s 500 Index - USA Market Focus'
            }
        }


if __name__ == "__main__":
    # Test USA market data loading
    print("Testing USA market data loading...")
    
    loader = USAMarketDataLoader()
    
    # Print market info
    print("\nUSA Market (Paper-compliant focus):")
    for code, info in loader.get_market_info().items():
        print(f"  {code}: {info['name']} ({info['country']}) - {info['symbol']}")
    
    # Test downloading a sample (shorter timeframe for testing)
    print("\nTesting sample data download (last 30 days)...")
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Test USA market
        sample_data = loader.download_market_data(
            'SP500', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        print("✅ USA market data loader working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


# Backward compatibility alias
MultiMarketDataLoader = USAMarketDataLoader