"""
Technical indicators computation using TA-Lib - Paper-compliant implementation.
Implements all 40 indicators specified in the paper exactly as described using TA-Lib library.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, falling back to manual calculations")


class TechnicalIndicatorsTA:
    """Computes technical indicators using TA-Lib library - Paper compliant."""
    
    # List of exactly 40 technical indicators as specified in the paper (same as reference [17])
    INDICATORS = [
        'BBANDS', 'WMA', 'EMA', 'DEMA', 'KAMA', 'MAMA', 'MIDPRICE', 'SAR', 
        'SMA', 'T3', 'TEMA', 'TRIMA', 'AD', 'ADOSC', 'OBV', 'MEDPRICE', 
        'TYPPRICE', 'WCLPRICE', 'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 
        'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MFI', 'MINUS_DI', 'MOM', 
        'PLUS_DI', 'PPO', 'ROC', 'RSI', 'STOCH', 'STOCHRSI', 'ULTOSC', 
        'WILLR', 'LOG_RETURN'
    ]
    
    def __init__(self):
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib is required for paper-compliant indicators. Please install it.")
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute exactly 40 technical indicators from OHLCV data using TA-Lib.
        
        This follows the exact specifications from the paper's pseudocode,
        implementing the same 40 indicators as reference [17].
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with exactly 40 computed features
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")
        
        features = {}
        
        # Extract price and volume arrays
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)
        
        print("Computing TA-Lib indicators exactly as specified in paper...")
        
        # Overlap Studies (as per paper pseudocode) - exactly 40 indicators
        upper, middle, lower = talib.BBANDS(close_prices)
        features['BBANDS'] = middle  # Use middle band as representative BBANDS value
        
        features['WMA'] = talib.WMA(close_prices)
        features['EMA'] = talib.EMA(close_prices)
        features['DEMA'] = talib.DEMA(close_prices)
        features['KAMA'] = talib.KAMA(close_prices)
        mama, fama = talib.MAMA(close_prices)
        features['MAMA'] = mama  # Use only MAMA, not FAMA
        features['MIDPRICE'] = talib.MIDPRICE(high_prices, low_prices)
        features['SAR'] = talib.SAR(high_prices, low_prices)
        features['SMA'] = talib.SMA(close_prices)
        features['T3'] = talib.T3(close_prices)
        features['TEMA'] = talib.TEMA(close_prices)
        features['TRIMA'] = talib.TRIMA(close_prices)
        
        # Volume indicators
        features['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
        features['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volume)
        features['OBV'] = talib.OBV(close_prices, volume)
        
        # Price transform
        features['MEDPRICE'] = talib.MEDPRICE(high_prices, low_prices)
        features['TYPPRICE'] = talib.TYPPRICE(high_prices, low_prices, close_prices)
        features['WCLPRICE'] = talib.WCLPRICE(high_prices, low_prices, close_prices)
        
        # Momentum indicators
        features['ADX'] = talib.ADX(high_prices, low_prices, close_prices)
        features['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices)
        features['APO'] = talib.APO(close_prices)
        
        aroondown, aroonup = talib.AROON(high_prices, low_prices)
        features['AROON'] = (aroondown + aroonup) / 2  # Use average of both components
        features['AROONOSC'] = talib.AROONOSC(high_prices, low_prices)
        
        features['BOP'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)
        features['CCI'] = talib.CCI(high_prices, low_prices, close_prices)
        features['CMO'] = talib.CMO(close_prices)
        features['DX'] = talib.DX(high_prices, low_prices, close_prices)
        
        macd, macd_signal, macd_hist = talib.MACD(close_prices)
        features['MACD'] = macd  # Use main MACD line as representative value
        
        features['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume)
        features['MINUS_DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
        features['MOM'] = talib.MOM(close_prices)
        features['PLUS_DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
        features['PPO'] = talib.PPO(close_prices)
        features['ROC'] = talib.ROC(close_prices)
        features['RSI'] = talib.RSI(close_prices)
        
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        features['STOCH'] = slowk  # Use %K as representative STOCH value
        
        # STOCHRSI can return different shapes - handle carefully
        try:
            stochrsi = talib.STOCHRSI(close_prices)
            if isinstance(stochrsi, tuple):
                features['STOCHRSI'] = stochrsi[0]  # Take first component
            else:
                features['STOCHRSI'] = stochrsi
        except Exception as e:
            print(f"Warning: STOCHRSI failed, using simplified version: {e}")
            # Fallback: simplified StochRSI calculation
            rsi_values = talib.RSI(close_prices)
            rsi_min = pd.Series(rsi_values).rolling(window=14, min_periods=1).min()
            rsi_max = pd.Series(rsi_values).rolling(window=14, min_periods=1).max()
            features['STOCHRSI'] = 100 * (rsi_values - rsi_min) / (rsi_max - rsi_min + 1e-10)
        features['ULTOSC'] = talib.ULTOSC(high_prices, low_prices, close_prices)
        features['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices)
        
        # Custom indicator: LOG_RETURN (as specified in paper)
        features['LOG_RETURN'] = np.log(close_prices / np.roll(close_prices, 1))
        features['LOG_RETURN'][0] = 0  # First value set to 0
        
        # Create DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        
        # Handle missing values (TA-Lib functions may produce NaN for initial periods)
        # Forward fill and backward fill to handle NaN values
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN values, fill with 0
        feature_df = feature_df.fillna(0)
        
        print(f"✓ Computed {len(feature_df.columns)} TA-Lib indicators for {len(feature_df)} samples")
        
        # Verify we have exactly 40 indicators as per paper
        if len(feature_df.columns) != 40:
            print(f"WARNING: Expected exactly 40 indicators, got {len(feature_df.columns)}")
            print(f"Indicators: {list(feature_df.columns)}")
        else:
            print("✓ Paper-compliant: Exactly 40 technical indicators computed")
        
        return feature_df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for compute_features to maintain compatibility."""
        return self.compute_features(df)
    
    def validate_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate computed features and return statistics."""
        stats = {
            'n_features': len(features_df.columns),
            'n_samples': len(features_df),
            'missing_values': features_df.isnull().sum().sum(),
            'inf_values': np.isinf(features_df.values).sum(),
            'feature_names': list(features_df.columns),
            'expected_features': len(self.INDICATORS),
            'features_match': set(features_df.columns) == set(self.INDICATORS)
        }
        
        print(f"Feature validation:")
        print(f"  Features computed: {stats['n_features']}")
        print(f"  Features expected: {stats['expected_features']}")
        print(f"  Features match paper: {stats['features_match']}")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Missing values: {stats['missing_values']}")
        print(f"  Infinite values: {stats['inf_values']}")
        
        if not stats['features_match']:
            missing = set(self.INDICATORS) - set(features_df.columns)
            extra = set(features_df.columns) - set(self.INDICATORS)
            if missing:
                print(f"  Missing indicators: {missing}")
            if extra:
                print(f"  Extra indicators: {extra}")
        
        return stats


if __name__ == "__main__":
    # Test the TA-Lib indicators computation
    print("Testing TA-Lib technical indicators computation...")
    
    if not TALIB_AVAILABLE:
        print("❌ TA-Lib not available - please install it first")
        exit(1)
    
    # Create synthetic OHLCV data
    np.random.seed(42)
    n_samples = 1000
    
    base_price = 100
    returns = np.random.normal(0.001, 0.02, n_samples)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    
    daily_range = np.random.uniform(0.005, 0.02, n_samples)
    high_prices = np.maximum(open_prices, close_prices) * (1 + daily_range/2)
    low_prices = np.minimum(open_prices, close_prices) * (1 - daily_range/2)
    
    volume = np.random.lognormal(15, 0.5, n_samples).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    print(f"Created synthetic OHLCV data: {df.shape}")
    
    # Test indicators
    indicators = TechnicalIndicatorsTA()
    features = indicators.compute_features(df)
    stats = indicators.validate_features(features)
    
    print(f"\n✓ Features computed: {len(features.columns)}")
    print(f"✓ Expected features (40+): {len(indicators.INDICATORS)}")
    print(f"✓ Feature matrix shape: {features.shape}")
    
    if stats['features_match']:
        print("✅ All paper-specified indicators computed successfully with TA-Lib!")
    else:
        print("❌ Feature mismatch detected")