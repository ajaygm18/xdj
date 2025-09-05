"""
Market Data Loading for USA and Indian Markets - Paper Compliant
Supports both USA market (S&P 500) and Indian market (Reliance Industries).
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class USAMarketDataLoader:
    """Loads data from USA market (S&P 500) - Paper compliant implementation."""
    
    # USA market focus - S&P 500
    USA_MARKET = {
        'SP500': '^GSPC',    # S&P 500 Index - Primary focus for paper compliance
    }
    
    def __init__(self):
        """Initialize USA market data loader."""
        pass
    
    def download_market_data(self, market_code: str = "SP500", start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download S&P 500 data (USA market) - Paper compliant.
        
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
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            print(f"✓ Downloaded {len(data)} samples from {data.index[0].date()} to {data.index[-1].date()}")
            print(f"✓ Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            
            return data[required_cols]
            
        except Exception as e:
            print(f"Error downloading {market_code} data: {e}")
            raise
    
    def get_market_info(self) -> Dict:
        """Get information about available markets."""
        return {
            'available_markets': list(self.USA_MARKET.keys()),
            'symbols': self.USA_MARKET,
            'default_market': 'SP500',
            'description': 'USA Market Data Loader (S&P 500 focus)'
        }


class IndianMarketDataLoader:
    """Loads data from Indian market (Reliance Industries) - focused implementation."""
    
    # Indian market focus - Reliance Industries
    INDIAN_MARKET = {
        'RELIANCE': 'RELIANCE.NS',    # Reliance Industries (NSE) - Primary focus
    }
    
    def __init__(self):
        """Initialize Indian market data loader."""
        pass
    
    def download_market_data(self, market_code: str = "RELIANCE", start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download Reliance Industries data (Indian market).
        
        Args:
            market_code: Market code (only 'RELIANCE' supported)
            start_date: Start date in 'YYYY-MM-DD' format (default: paper start)
            end_date: End date in 'YYYY-MM-DD' format (default: paper end)
            
        Returns:
            DataFrame with OHLCV data
        """
        if market_code not in self.INDIAN_MARKET:
            print(f"Warning: Only RELIANCE (Indian) market supported. Using RELIANCE instead of {market_code}")
            market_code = "RELIANCE"
        
        symbol = self.INDIAN_MARKET[market_code]
        print(f"Downloading Indian market data ({symbol}) from {start_date} to {end_date}...")
        
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
            
            print(f"✓ Downloaded {len(data)} samples for Indian market (RELIANCE)")
            return data[required_cols]
            
        except Exception as e:
            print(f"❌ Failed to download Indian market data: {e}")
            raise
    
    def download_indian_market(self, start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download Indian market data (Reliance Industries) with paper-compliant timeframe.
        
        Args:
            start_date: Start date (paper default: 2005-01-01)
            end_date: End date (paper default: 2022-03-31)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading Indian market (Reliance Industries) data from {start_date} to {end_date}...")
        return self.download_market_data("RELIANCE", start_date, end_date)
    
    def get_market_info(self) -> Dict[str, Dict]:
        """Get information about Indian market (Reliance)."""
        return {
            'RELIANCE': {
                'name': 'Reliance Industries Limited',
                'country': 'India',
                'symbol': 'RELIANCE.NS',
                'description': 'Reliance Industries - Indian Market Focus'
            }
        }


if __name__ == "__main__":
    # Test Indian market data loading
    print("Testing Indian market data loading...")
    
    loader = IndianMarketDataLoader()
    
    # Print market info
    print("\nIndian Market (Paper-compliant focus):")
    for code, info in loader.get_market_info().items():
        print(f"  {code}: {info['name']} ({info['country']}) - {info['symbol']}")
    
    # Test downloading a sample (shorter timeframe for testing)
    print("\nTesting sample data download (last 30 days)...")
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Test Indian market
        sample_data = loader.download_market_data(
            'RELIANCE', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        print("✅ Indian market data loader working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


# Backward compatibility aliases
# Remove compatibility alias since we now have separate classes
# USAMarketDataLoader = IndianMarketDataLoader  # For compatibility
MultiMarketDataLoader = IndianMarketDataLoader