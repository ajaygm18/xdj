"""
Main execution script for PLSTM-TAL stock market prediction.
Implements the complete pipeline from data loading to model evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import argparse
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import modules that work without external dependencies
from model_plstm_tal import PLSTM_TAL
from baselines import BaselineModelFactory
from cae import CAEFeatureExtractor
from train import DataPreprocessor, ModelTrainer
from eval import ModelEvaluator
from indicators import TechnicalIndicators
from eemd import EEMDDenoiser


class SimplifiedPipeline:
    """Simplified pipeline that works without external dependencies."""
    
    def __init__(self, config: dict):
        """Initialize pipeline with configuration."""
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.results_dir = config.get('results_dir', 'results')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.evaluator = ModelEvaluator(save_plots=True, plot_dir=self.results_dir)
        
        # Import data loader
        from data_loader import YFinanceDataLoader
        self.data_loader = YFinanceDataLoader()
        
    def load_real_market_data(self, symbol: str = "^GSPC", years: int = 15) -> tuple:
        """Load real market data from Yahoo Finance."""
        print(f"Loading real market data for {symbol} ({years} years)...")
        
        # Calculate date range
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Download data
        self.data_loader.symbol = symbol
        raw_data = self.data_loader.download_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Get OHLCV data
        price_df = self.data_loader.get_ohlcv_data()
        
        # Generate technical indicators using proper implementation
        features_df = self.generate_proper_features(price_df)
        
        # Apply proper EEMD filtering with balanced parameters
        print("Applying optimized EEMD filtering...")
        denoiser = EEMDDenoiser(n_ensembles=50, noise_scale=0.2, w=7)  # Reduced from 100 to 50
        filtered_prices, eemd_metadata = denoiser.process_price_series(price_df['close'])
        
        print(f"Real data loaded: {len(price_df)} samples from {price_df.index[0].date()} to {price_df.index[-1].date()}")
        print(f"Price range: ${price_df['close'].min():.2f} - ${price_df['close'].max():.2f}")
        print(f"EEMD metadata: {eemd_metadata}")
        
        return features_df, price_df['close'], filtered_prices
        
    def generate_synthetic_data(self, n_samples: int = 4000) -> tuple:
        """Generate realistic synthetic stock market data."""
        print("Generating synthetic S&P 500 data...")
        
        np.random.seed(42)  # For reproducibility
        
        # Generate dates (business days only)
        start_date = pd.Timestamp('2005-01-01')
        dates = pd.bdate_range(start=start_date, periods=n_samples)
        
        # Generate realistic price series
        initial_price = 1200.0
        daily_return_mean = 0.0008  # ~20% annual return
        daily_return_std = 0.012    # ~19% annual volatility
        
        returns = np.random.normal(daily_return_mean, daily_return_std, n_samples)
        
        # Add some autocorrelation
        for i in range(1, len(returns)):
            returns[i] += 0.05 * returns[i-1]
        
        # Calculate cumulative prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        close_prices = prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = initial_price
        
        # Add some noise to opens
        open_gap = np.random.normal(0, 0.002, n_samples)
        open_prices = open_prices * (1 + open_gap)
        
        # High and low prices
        daily_range = np.random.uniform(0.005, 0.025, n_samples)
        high_prices = np.maximum(open_prices, close_prices) * (1 + daily_range/2)
        low_prices = np.minimum(open_prices, close_prices) * (1 - daily_range/2)
        
        # Volume
        base_volume = 50_000_000
        volume_volatility = np.random.lognormal(0, 0.5, n_samples)
        volume = (base_volume * volume_volatility).astype(int)
        
        # Create price DataFrame
        price_df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        # Generate technical indicators using proper implementation
        features_df = self.generate_proper_features(price_df)
        
        # Apply proper EEMD filtering
        filtered_prices = self.apply_eemd_filtering(price_df['close'])
        
        print(f"Generated {len(price_df)} samples from {dates[0].date()} to {dates[-1].date()}")
        print(f"Price range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
        
        return features_df, price_df['close'], filtered_prices
    
    def generate_proper_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Generate proper technical indicators using paper-compliant implementation."""
        print("Computing 40+ technical indicators as specified in paper...")
        
        indicators = TechnicalIndicators()
        features_df = indicators.compute_features(price_df)
        
        print(f"Generated {len(features_df.columns)} paper-compliant technical indicators")
        return features_df
    
    def generate_simple_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Generate simplified technical indicators without TA-Lib."""
        features = {}
        
        close = price_df['close']
        high = price_df['high']
        low = price_df['low']
        volume = price_df['volume']
        
        # Simple moving averages
        for period in [5, 10, 20, 50]:
            features[f'SMA_{period}'] = close.rolling(period).mean()
            features[f'EMA_{period}'] = close.ewm(span=period).mean()
        
        # Price-based features
        features['LOG_RETURN'] = np.log(close / close.shift(1))
        features['PRICE_CHANGE'] = close.pct_change()
        features['HIGH_LOW_RATIO'] = high / low
        features['CLOSE_OPEN_RATIO'] = close / price_df['open']
        
        # Volatility measures
        for period in [5, 10, 20]:
            features[f'VOLATILITY_{period}'] = close.pct_change().rolling(period).std()
            features[f'PRICE_RANGE_{period}'] = (high - low).rolling(period).mean()
        
        # Volume features
        features['VOLUME_SMA_10'] = volume.rolling(10).mean()
        features['VOLUME_RATIO'] = volume / volume.rolling(20).mean()
        
        # Momentum indicators
        for period in [5, 10, 14, 20]:
            features[f'MOMENTUM_{period}'] = close / close.shift(period) - 1
            features[f'ROC_{period}'] = close.pct_change(period)
        
        # Simple RSI approximation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands approximation
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features['BB_UPPER'] = sma_20 + (2 * std_20)
        features['BB_LOWER'] = sma_20 - (2 * std_20)
        features['BB_WIDTH'] = features['BB_UPPER'] - features['BB_LOWER']
        
        # MACD approximation
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['MACD'] = ema_12 - ema_26
        features['MACD_SIGNAL'] = features['MACD'].ewm(span=9).mean()
        features['MACD_HIST'] = features['MACD'] - features['MACD_SIGNAL']
        
        # Create DataFrame and drop NaN values
        features_df = pd.DataFrame(features, index=price_df.index)
        features_df = features_df.dropna()
        
        print(f"Generated {len(features_df.columns)} simplified technical indicators")
        return features_df
    
    def apply_eemd_filtering(self, prices: pd.Series) -> pd.Series:
        """Apply proper EEMD filtering using enhanced algorithm."""
        print("Applying proper EEMD filtering with full algorithm implementation...")
        
        # Use the optimized EEMD denoiser with fewer ensembles for efficiency
        denoiser = EEMDDenoiser(n_ensembles=50, noise_scale=0.2, w=7)  # Balanced parameters
        filtered_prices, metadata = denoiser.process_price_series(prices)
        
        print(f"Enhanced EEMD filtering completed with {metadata['n_imfs']} IMFs")
        return filtered_prices
    
    def simulate_eemd_filtering(self, prices: pd.Series) -> pd.Series:
        """Simulate EEMD filtering with simple noise removal."""
        # Apply simple smoothing as EEMD approximation
        # Rolling median to remove high-frequency noise
        window = 5
        filtered = prices.rolling(window, center=True).median()
        
        # Fill NaN values
        filtered = filtered.fillna(method='bfill').fillna(method='ffill')
        
        print("Applied simplified EEMD filtering (rolling median)")
        return filtered
    
    def run_experiment(self):
        """Run the complete experiment pipeline with real data and optimized parameters."""
        print("=" * 60)
        print("PLSTM-TAL Stock Market Prediction Pipeline - REAL DATA")
        print("=" * 60)
        
        # Step 1: Load Real Market Data (reduced timeframe for efficiency)
        symbol = self.config.get('symbol', '^GSPC')
        years = self.config.get('data_years', 10)  # Reduced from 15 to 10 years
        features_df, prices, filtered_prices = self.load_real_market_data(symbol, years)
        
        # Step 2: Train CAE for Feature Extraction with optimized parameters
        print("\n--- Training Contractive Autoencoder (Optimized) ---")
        cae_config = self.config.get('cae', {})
        cae = CAEFeatureExtractor(
            hidden_dim=cae_config.get('hidden_dim', 128),  # Increased from 64
            encoding_dim=cae_config.get('encoding_dim', 32),  # Increased from 16
            dropout=cae_config.get('dropout', 0.2),  # Increased from 0.1
            lambda_reg=cae_config.get('lambda_reg', 5e-4)  # Increased from 1e-4
        )
        
        cae_history = cae.train(
            features_df, 
            epochs=cae_config.get('epochs', 200),  # Increased from 50
            batch_size=cae_config.get('batch_size', 128),  # Increased from 64
            learning_rate=cae_config.get('learning_rate', 5e-4),  # Decreased for stability
            verbose=True
        )
        
        # Step 3: Prepare Data for Model Training
        print("\n--- Preparing Training Data (Optimized) ---")
        train_config = self.config.get('training', {})
        preprocessor = DataPreprocessor(
            window_length=train_config.get('window_length', 30),  # Increased from 20
            step_size=train_config.get('step_size', 1)
        )
        
        X_sequences, y_labels = preprocessor.prepare_data(
            features_df, prices, cae, filtered_prices
        )
        
        # Train/validation/test split (70/15/15 for more training data)
        from sklearn.model_selection import train_test_split
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_sequences, y_labels, test_size=0.15, random_state=42, stratify=y_labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 of 0.85 ≈ 0.15
        )
        
        print(f"Data splits (optimized):")
        print(f"  Training: {X_train.shape}")
        print(f"  Validation: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        # Step 4: Train Models with Optimized Hyperparameters
        print("\n--- Training Models (Full Optimization) ---")
        models = {}
        results = {}
        
        input_size = X_train.shape[2]
        
        # PLSTM-TAL (Main model) - Optimized Architecture
        print("\nTraining PLSTM-TAL with optimized parameters...")
        plstm_config = self.config.get('plstm_tal', {})
        plstm_model = PLSTM_TAL(
            input_size=input_size,
            hidden_size=plstm_config.get('hidden_size', 128),  # Increased from 64
            num_layers=plstm_config.get('num_layers', 2),  # Increased from 1
            dropout=plstm_config.get('dropout', 0.3),  # Increased from 0.1
            activation='tanh'
        )
        
        plstm_trainer = ModelTrainer(plstm_model)
        plstm_history = plstm_trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=train_config.get('epochs', 300),  # Increased from 100
            batch_size=train_config.get('batch_size', 64),  # Increased from 32
            learning_rate=train_config.get('learning_rate', 1e-4),  # Decreased for stability
            optimizer_name='adamax',
            early_stopping_patience=train_config.get('patience', 30)  # Increased patience
        )
        models['PLSTM-TAL'] = plstm_model
        
        # Baseline Models with Optimized Parameters
        baseline_config = self.config.get('baselines', {})
        
        # CNN - Enhanced Architecture
        print("\nTraining CNN baseline (optimized)...")
        cnn_model = BaselineModelFactory.create_model(
            'cnn', input_size, X_train.shape[1],
            num_filters=baseline_config.get('cnn_filters', 64),  # Increased from 32
            dropout=baseline_config.get('dropout', 0.3)  # Increased from 0.1
        )
        cnn_trainer = ModelTrainer(cnn_model)
        cnn_trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=baseline_config.get('epochs', 200),  # Increased from 50
            batch_size=64, learning_rate=5e-4, optimizer_name='adam'  # Optimized
        )
        models['CNN'] = cnn_model
        
        # LSTM - Enhanced Architecture
        print("\nTraining LSTM baseline (optimized)...")
        lstm_model = BaselineModelFactory.create_model(
            'lstm', input_size,
            hidden_size=baseline_config.get('lstm_hidden', 64),  # Increased from 32
            dropout=baseline_config.get('dropout', 0.3)  # Increased from 0.1
        )
        lstm_trainer = ModelTrainer(lstm_model)
        lstm_trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=baseline_config.get('epochs', 200),  # Increased from 50
            batch_size=64, learning_rate=5e-4, optimizer_name='adam'  # Optimized
        )
        models['LSTM'] = lstm_model
        
        # SVM - Optimized Parameters
        print("\nTraining SVM baseline (optimized)...")
        svm_model = BaselineModelFactory.create_model('svm', input_size, 
                                                     C=baseline_config.get('svm_C', 10.0),  # Increased regularization
                                                     gamma=baseline_config.get('svm_gamma', 'scale'))
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        
        # Random Forest - Optimized Parameters
        print("\nTraining Random Forest baseline (optimized)...")
        rf_model = BaselineModelFactory.create_model(
            'rf', input_size,
            n_estimators=baseline_config.get('rf_trees', 200),  # Increased from 100
            max_depth=baseline_config.get('rf_depth', 15),  # Added depth control
            min_samples_split=baseline_config.get('rf_min_split', 10)  # Added split control
        )
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        
        # Step 5: Evaluate Models
        print("\n--- Evaluating Models ---")
        
        # Evaluate each model
        for name, model in models.items():
            is_torch = isinstance(model, torch.nn.Module)
            result = self.evaluator.evaluate_model(
                model, X_test, y_test, name, is_torch_model=is_torch
            )
            results[name] = result
        
        # Compare models
        print("\n--- Model Comparison ---")
        comparison_df = self.evaluator.compare_models(results)
        print(comparison_df.to_string(index=False))
        
        # Save comparison plot
        self.evaluator.plot_model_comparison(results)
        
        # Step 6: Save Results
        print("\n--- Saving Results ---")
        
        # Save metrics to JSON
        metrics_dict = {}
        for name, result in results.items():
            metrics_dict[name] = result['metrics']
        
        with open(f"{self.results_dir}/metrics.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        # Save comparison DataFrame
        comparison_df.to_csv(f"{self.results_dir}/model_comparison.csv", index=False)
        
        # Save model weights (PyTorch models only)
        for name, model in models.items():
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), f"{self.results_dir}/{name.lower()}_weights.pth")
        
        # Save CAE model
        cae.save_model(f"{self.results_dir}/cae_model.pth")
        
        # Save data information
        data_info = {
            'symbol': symbol,
            'years': years,
            'total_samples': len(features_df),
            'features_count': len(features_df.columns),
            'data_range': {
                'start': str(features_df.index[0].date()),
                'end': str(features_df.index[-1].date())
            },
            'price_statistics': {
                'min': float(prices.min()),
                'max': float(prices.max()),
                'mean': float(prices.mean()),
                'std': float(prices.std())
            }
        }
        
        with open(f"{self.results_dir}/data_info.json", 'w') as f:
            json.dump(data_info, f, indent=2)
        
        print(f"\nResults saved to {self.results_dir}/")
        print("Pipeline completed successfully!")
        
        return results, comparison_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='PLSTM-TAL Stock Market Prediction')
    parser.add_argument('--config', type=str, default='config.json',
                      help='Configuration file path')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    
    args = parser.parse_args()
    
    # Default configuration - Optimized for target accuracy
    default_config = {
        "data_dir": "data",
        "results_dir": args.output,
        "symbol": "^GSPC",
        "data_years": 10,
        "cae": {
            "hidden_dim": 128,
            "encoding_dim": 32,
            "dropout": 0.2,
            "lambda_reg": 5e-4,
            "epochs": 200,
            "batch_size": 128,
            "learning_rate": 5e-4
        },
        "training": {
            "window_length": 30,
            "step_size": 1,
            "epochs": 300,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "patience": 30
        },
        "plstm_tal": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3
        },
        "baselines": {
            "epochs": 200,
            "dropout": 0.3,
            "cnn_filters": 64,
            "lstm_hidden": 64,
            "rf_trees": 200,
            "rf_depth": 15,
            "rf_min_split": 10,
            "svm_C": 10.0,
            "svm_gamma": "scale"
        }
    }
    
    # Load configuration if exists
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update defaults with loaded config
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    else:
        print(f"Configuration file {args.config} not found, using defaults")
        # Save default config
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default configuration saved to {args.config}")
    
    config = default_config
    
    # Run pipeline
    pipeline = SimplifiedPipeline(config)
    results, comparison = pipeline.run_experiment()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()