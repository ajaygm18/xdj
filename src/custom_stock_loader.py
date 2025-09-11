"""
Custom Stock Data Loader - Universal Yahoo Finance Stock Support
Supports loading any valid Yahoo Finance stock symbol with the same data preprocessing pipeline.
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CustomStockDataLoader:
    """Loads data for any Yahoo Finance stock symbol."""
    
    def __init__(self):
        """Initialize custom stock data loader."""
        self.cache = {}  # Simple cache for downloaded data
    
    def download_stock_data(self, symbol: str, start_date: str = "2005-01-01", 
                           end_date: str = "2022-03-31") -> pd.DataFrame:
        """
        Download stock data for any Yahoo Finance symbol.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., 'AAPL', 'GOOGL', 'TSLA', '^GSPC', 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format (default: paper start)
            end_date: End date in 'YYYY-MM-DD' format (default: paper end)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # Check cache first
        if cache_key in self.cache:
            print(f"Using cached data for {symbol}")
            return self.cache[cache_key]
        
        print(f"Downloading {symbol} data from {start_date} to {end_date}...")
        
        try:
            # Clean the symbol (remove any extra spaces)
            symbol = symbol.strip().upper()
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol '{symbol}'. Please check if the symbol is valid.")
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Clean data (remove any NaN values)
            initial_len = len(data)
            data = data.dropna()
            final_len = len(data)
            
            if initial_len > final_len:
                print(f"Removed {initial_len - final_len} rows with missing data")
            
            if len(data) < 100:
                raise ValueError(f"Insufficient data for {symbol}: only {len(data)} trading days available")
            
            # Validate price data
            self._validate_price_data(data, symbol)
            
            print(f"✓ Downloaded {len(data)} samples from {data.index[0].date()} to {data.index[-1].date()}")
            
            # Determine currency based on symbol patterns
            currency = self._detect_currency(symbol)
            print(f"✓ Price range: {currency}{data['close'].min():.2f} - {currency}{data['close'].max():.2f}")
            
            # Cache the result
            self.cache[cache_key] = data[required_cols]
            
            return data[required_cols]
            
        except Exception as e:
            print(f"Error downloading {symbol} data: {e}")
            raise
    
    def _validate_price_data(self, data: pd.DataFrame, symbol: str) -> None:
        """Validate the downloaded price data for basic sanity checks."""
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    print(f"Warning: {negative_count} non-positive values in {col} for {symbol}")
        
        # Check for impossible OHLC relationships
        if all(col in data.columns for col in price_cols):
            # High should be >= max(open, close, low)
            invalid_high = ((data['high'] < data['open']) | 
                           (data['high'] < data['close']) |
                           (data['high'] < data['low'])).sum()
            if invalid_high > 0:
                print(f"Warning: {invalid_high} rows with invalid high prices for {symbol}")
            
            # Low should be <= min(open, close, high)
            invalid_low = ((data['low'] > data['open']) | 
                          (data['low'] > data['close']) |
                          (data['low'] > data['high'])).sum()
            if invalid_low > 0:
                print(f"Warning: {invalid_low} rows with invalid low prices for {symbol}")
    
    def _detect_currency(self, symbol: str) -> str:
        """Detect currency symbol based on stock exchange."""
        symbol = symbol.upper()
        
        # Indian exchanges
        if any(suffix in symbol for suffix in ['.NS', '.BO']):
            return "₹"
        
        # European exchanges
        if any(suffix in symbol for suffix in ['.PA', '.DE', '.L']):
            return "€" if '.PA' in symbol or '.DE' in symbol else "£"
        
        # Japanese exchanges
        if '.T' in symbol:
            return "¥"
        
        # Canadian exchanges
        if '.TO' in symbol or '.V' in symbol:
            return "C$"
        
        # Default to USD for US markets and others
        return "$"
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get basic information about a stock symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'market_cap': info.get('marketCap', 'Unknown')
            }
        except Exception as e:
            print(f"Could not fetch info for {symbol}: {e}")
            return {
                'symbol': symbol,
                'name': 'Unknown',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'country': 'Unknown',
                'currency': 'USD',
                'exchange': 'Unknown',
                'market_cap': 'Unknown'
            }
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate if a symbol exists on Yahoo Finance.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            symbol = symbol.strip().upper()
            ticker = yf.Ticker(symbol)
            
            # Try to get some basic info
            info = ticker.info
            
            # Try to get a small amount of recent data
            test_data = ticker.history(period="5d")
            
            if test_data.empty:
                return False, f"No trading data found for symbol '{symbol}'"
            
            return True, f"Valid symbol: {info.get('longName', symbol)}"
            
        except Exception as e:
            return False, f"Invalid symbol '{symbol}': {str(e)}"
    
    def get_popular_symbols(self) -> Dict[str, list]:
        """Get a list of popular stock symbols by category."""
        return {
            "US Large Cap": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"
            ],
            "US Indices": [
                "^GSPC", "^DJI", "^IXIC", "^RUT"
            ],
            "Indian Stocks": [
                "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS"
            ],
            "European Stocks": [
                "ASML.AS", "SAP.DE", "NESN.SW", "NOVO.CO", "MC.PA"
            ],
            "Crypto": [
                "BTC-USD", "ETH-USD", "ADA-USD"
            ],
            "Commodities": [
                "GC=F", "CL=F", "SI=F"
            ]
        }
    
    def clear_cache(self):
        """Clear the download cache."""
        self.cache.clear()
        print("Download cache cleared")


if __name__ == "__main__":
    # Test the custom stock loader
    print("Testing Custom Stock Data Loader...")
    
    loader = CustomStockDataLoader()
    
    # Test with different types of symbols
    test_symbols = ["AAPL", "TSLA", "^GSPC", "RELIANCE.NS"]
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Validate symbol
        is_valid, message = loader.validate_symbol(symbol)
        print(f"Validation: {message}")
        
        if is_valid:
            # Get stock info
            info = loader.get_stock_info(symbol)
            print(f"Name: {info['name']}")
            print(f"Country: {info['country']}")
            print(f"Exchange: {info['exchange']}")
            
            # Test download (small period for testing)
            try:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                data = loader.download_stock_data(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                )
                print(f"Downloaded {len(data)} days of data")
            except Exception as e:
                print(f"Download failed: {e}")
    
    print("\n--- Popular Symbols ---")
    popular = loader.get_popular_symbols()
    for category, symbols in popular.items():
        print(f"{category}: {', '.join(symbols[:5])}...")