"""
Technical indicators computation using TA-Lib.
Implements all 40 indicators specified in the paper plus LOG_RETURN.
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, Any


class TechnicalIndicators:
    """Computes technical indicators using TA-Lib."""
    
    # List of 40 technical indicators as specified in the paper
    INDICATORS = [
        'BBANDS', 'WMA', 'EMA', 'DEMA', 'KAMA', 'MAMA', 'MIDPRICE', 'SAR', 
        'SMA', 'T3', 'TEMA', 'TRIMA', 'AD', 'ADOSC', 'OBV', 'MEDPRICE', 
        'TYPPRICE', 'WCLPRICE', 'ADX', 'ADXR', 'APO', 'AROON', 'AROONOSC', 
        'BOP', 'CCI', 'CMO', 'DX', 'MACD', 'MFI', 'MINUS_DI', 'MOM', 
        'PLUS_DI', 'LOG_RETURN', 'PPO', 'ROC', 'RSI', 'STOCH', 'STOCHRSI', 
        'ULTOSC', 'WILLR'
    ]
    
    def __init__(self):
        pass
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with all computed features
        """
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("DataFrame must contain OHLCV columns")
        
        features = {}
        
        # Extract price and volume arrays
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values
        
        # Overlap Studies
        try:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            features['BBANDS_upper'] = upper
            features['BBANDS_middle'] = middle
            features['BBANDS_lower'] = lower
        except:
            features['BBANDS_upper'] = np.full_like(close_prices, np.nan)
            features['BBANDS_middle'] = np.full_like(close_prices, np.nan)
            features['BBANDS_lower'] = np.full_like(close_prices, np.nan)
        
        features['WMA'] = talib.WMA(close_prices, timeperiod=30)
        features['EMA'] = talib.EMA(close_prices, timeperiod=30)
        features['DEMA'] = talib.DEMA(close_prices, timeperiod=30)
        features['KAMA'] = talib.KAMA(close_prices, timeperiod=30)
        
        try:
            mama, fama = talib.MAMA(close_prices, fastlimit=0.5, slowlimit=0.05)
            features['MAMA'] = mama
            features['FAMA'] = fama
        except:
            features['MAMA'] = np.full_like(close_prices, np.nan)
            features['FAMA'] = np.full_like(close_prices, np.nan)
        
        features['MIDPRICE'] = talib.MIDPRICE(high_prices, low_prices, timeperiod=14)
        features['SAR'] = talib.SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2)
        features['SMA'] = talib.SMA(close_prices, timeperiod=30)
        features['T3'] = talib.T3(close_prices, timeperiod=5, vfactor=0.7)
        features['TEMA'] = talib.TEMA(close_prices, timeperiod=30)
        features['TRIMA'] = talib.TRIMA(close_prices, timeperiod=30)
        
        # Volume Indicators
        features['AD'] = talib.AD(high_prices, low_prices, close_prices, volume)
        features['ADOSC'] = talib.ADOSC(high_prices, low_prices, close_prices, volume, fastperiod=3, slowperiod=10)
        features['OBV'] = talib.OBV(close_prices, volume)
        
        # Price Transform
        features['MEDPRICE'] = talib.MEDPRICE(high_prices, low_prices)
        features['TYPPRICE'] = talib.TYPPRICE(high_prices, low_prices, close_prices)
        features['WCLPRICE'] = talib.WCLPRICE(high_prices, low_prices, close_prices)
        
        # Momentum Indicators
        features['ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
        features['ADXR'] = talib.ADXR(high_prices, low_prices, close_prices, timeperiod=14)
        features['APO'] = talib.APO(close_prices, fastperiod=12, slowperiod=26, matype=0)
        
        try:
            aroondown, aroonup = talib.AROON(high_prices, low_prices, timeperiod=14)
            features['AROON_down'] = aroondown
            features['AROON_up'] = aroonup
        except:
            features['AROON_down'] = np.full_like(close_prices, np.nan)
            features['AROON_up'] = np.full_like(close_prices, np.nan)
        
        features['AROONOSC'] = talib.AROONOSC(high_prices, low_prices, timeperiod=14)
        features['BOP'] = talib.BOP(open_prices, high_prices, low_prices, close_prices)
        features['CCI'] = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
        features['CMO'] = talib.CMO(close_prices, timeperiod=14)
        features['DX'] = talib.DX(high_prices, low_prices, close_prices, timeperiod=14)
        
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            features['MACD'] = macd
            features['MACD_signal'] = macd_signal
            features['MACD_hist'] = macd_hist
        except:
            features['MACD'] = np.full_like(close_prices, np.nan)
            features['MACD_signal'] = np.full_like(close_prices, np.nan)
            features['MACD_hist'] = np.full_like(close_prices, np.nan)
        
        features['MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=14)
        features['MINUS_DI'] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        features['MOM'] = talib.MOM(close_prices, timeperiod=10)
        features['PLUS_DI'] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=14)
        features['PPO'] = talib.PPO(close_prices, fastperiod=12, slowperiod=26, matype=0)
        features['ROC'] = talib.ROC(close_prices, timeperiod=10)
        features['RSI'] = talib.RSI(close_prices, timeperiod=14)
        
        try:
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, 
                                     fastk_period=5, slowk_period=3, slowk_matype=0, 
                                     slowd_period=3, slowd_matype=0)
            features['STOCH_k'] = slowk
            features['STOCH_d'] = slowd
        except:
            features['STOCH_k'] = np.full_like(close_prices, np.nan)
            features['STOCH_d'] = np.full_like(close_prices, np.nan)
        
        features['STOCHRSI'] = talib.STOCHRSI(close_prices, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0]
        features['ULTOSC'] = talib.ULTOSC(high_prices, low_prices, close_prices, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        features['WILLR'] = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # Custom indicator: LOG_RETURN
        features['LOG_RETURN'] = np.log(close_prices / np.roll(close_prices, 1))
        features['LOG_RETURN'][0] = np.nan  # First value is undefined
        
        # Create DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        
        # Drop rows with NaN values
        feature_df = feature_df.dropna()
        
        print(f"Computed {len(feature_df.columns)} features for {len(feature_df)} samples")
        return feature_df
    
    def validate_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate computed features and return statistics."""
        stats = {
            'n_features': len(features_df.columns),
            'n_samples': len(features_df),
            'missing_values': features_df.isnull().sum().sum(),
            'inf_values': np.isinf(features_df.values).sum(),
            'feature_names': list(features_df.columns)
        }
        
        print(f"Feature validation:")
        print(f"  Features: {stats['n_features']}")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Missing values: {stats['missing_values']}")
        print(f"  Infinite values: {stats['inf_values']}")
        
        return stats


if __name__ == "__main__":
    # Test the indicators computation
    import sys
    sys.path.append('..')
    from io import DataLoader
    
    loader = DataLoader()
    df = loader.download_sp500_data()
    
    indicators = TechnicalIndicators()
    features = indicators.compute_features(df)
    stats = indicators.validate_features(features)
    
    print(f"\nFeatures computed: {list(features.columns)}")
    print(f"Feature matrix shape: {features.shape}")