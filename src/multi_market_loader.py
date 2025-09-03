"""
Multi-market data loading support for the four markets mentioned in the paper:
- S&P 500 (U.S.)
- FTSE 100 (U.K.) 
- SSE Composite (China)
- Nifty 50 (India)
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiMarketDataLoader:
    """Loads data from multiple stock markets as specified in the paper."""
    
    # Market symbols as mentioned in the paper
    PAPER_MARKETS = {
        'SP500': '^GSPC',      # S&P 500 (U.S.)
        'FTSE': '^FTSE',       # FTSE 100 (U.K.)
        'SSE': '000001.SS',    # SSE Composite (China)
        'NIFTY50': '^NSEI'     # Nifty 50 (India)
    }
    
    def __init__(self):
        pass
    
    def download_market_data(self, market_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download data for a specific market.
        
        Args:
            market_code: Market code ('SP500', 'FTSE', 'SSE', 'NIFTY50')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        if market_code not in self.PAPER_MARKETS:
            raise ValueError(f"Unknown market code: {market_code}. Use one of: {list(self.PAPER_MARKETS.keys())}")
        
        symbol = self.PAPER_MARKETS[market_code]
        print(f"Downloading {market_code} data ({symbol}) from {start_date} to {end_date}...")
        
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
            
            print(f"✓ Downloaded {len(data)} samples for {market_code}")
            return data[required_cols]
            
        except Exception as e:
            print(f"❌ Failed to download {market_code} data: {e}")
            raise
    
    def download_all_markets(self, start_date: str = "2005-01-01", end_date: str = "2022-03-31") -> Dict[str, pd.DataFrame]:
        """
        Download data for all four markets mentioned in the paper.
        
        Args:
            start_date: Start date (paper default: 2005-01-01)
            end_date: End date (paper default: 2022-03-31)
            
        Returns:
            Dictionary mapping market codes to DataFrames
        """
        print(f"Downloading data for all paper markets from {start_date} to {end_date}...")
        
        market_data = {}
        for market_code in self.PAPER_MARKETS:
            try:
                data = self.download_market_data(market_code, start_date, end_date)
                market_data[market_code] = data
            except Exception as e:
                print(f"Warning: Failed to download {market_code}: {e}")
                # Continue with other markets
        
        print(f"Successfully downloaded data for {len(market_data)} markets")
        return market_data
    
    def get_market_info(self) -> Dict[str, Dict]:
        """Get information about all markets."""
        return {
            'SP500': {
                'name': 'S&P 500',
                'country': 'United States',
                'symbol': '^GSPC',
                'description': 'Standard & Poor\'s 500 Index'
            },
            'FTSE': {
                'name': 'FTSE 100',
                'country': 'United Kingdom', 
                'symbol': '^FTSE',
                'description': 'Financial Times Stock Exchange 100 Index'
            },
            'SSE': {
                'name': 'SSE Composite',
                'country': 'China',
                'symbol': '000001.SS', 
                'description': 'Shanghai Stock Exchange Composite Index'
            },
            'NIFTY50': {
                'name': 'Nifty 50',
                'country': 'India',
                'symbol': '^NSEI',
                'description': 'National Stock Exchange of India Fifty Index'
            }
        }


if __name__ == "__main__":
    # Test multi-market data loading
    print("Testing multi-market data loading...")
    
    loader = MultiMarketDataLoader()
    
    # Print market info
    print("\nPaper-specified markets:")
    for code, info in loader.get_market_info().items():
        print(f"  {code}: {info['name']} ({info['country']}) - {info['symbol']}")
    
    # Test downloading a sample (shorter timeframe for testing)
    print("\nTesting sample data download (last 30 days)...")
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    try:
        # Test one market first
        sample_data = loader.download_market_data(
            'SP500', 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        print("✅ Multi-market data loader working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")