"""
Test script to verify the integration of custom stock loading with existing model pipeline.
"""
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from src.custom_stock_loader import CustomStockDataLoader
from src.indicators import TechnicalIndicators
from src.model_plstm_tal import PLSTM_TAL
from src.cae import CAEFeatureExtractor
from src.train import DataPreprocessor, ModelTrainer
from src.eval import ModelEvaluator
from src.eemd import EEMDDenoiser

def test_integration():
    """Test the complete integration pipeline."""
    print("🧪 Testing Custom Stock Pipeline Integration")
    print("=" * 50)
    
    # Step 1: Load custom stock data
    print("\n1. Testing custom stock data loading...")
    loader = CustomStockDataLoader()
    
    try:
        # Use a reliable symbol with sufficient data
        symbol = "AAPL"
        stock_data = loader.download_stock_data(symbol, "2020-01-01", "2021-01-01")
        print(f"✅ Successfully loaded {len(stock_data)} days of {symbol} data")
    except Exception as e:
        print(f"❌ Stock data loading failed: {e}")
        return False
    
    # Step 2: Test technical indicators
    print("\n2. Testing technical indicators...")
    try:
        indicators = TechnicalIndicators()
        features_df = indicators.compute_features(stock_data)
        print(f"✅ Generated {len(features_df.columns)} technical indicators")
    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        return False
    
    # Step 3: Test EEMD (simplified)
    print("\n3. Testing EEMD denoising...")
    try:
        denoiser = EEMDDenoiser(n_ensembles=10, noise_scale=0.2, w=7)  # Reduced for testing
        filtered_prices, metadata = denoiser.process_price_series(stock_data['close'])
        print(f"✅ EEMD completed with {metadata['n_imfs']} IMFs")
    except Exception as e:
        print(f"❌ EEMD failed: {e}")
        return False
    
    # Step 4: Test CAE
    print("\n4. Testing Contractive Autoencoder...")
    try:
        cae = CAEFeatureExtractor(hidden_dim=32, encoding_dim=8, dropout=0.1, lambda_reg=1e-4)
        cae_history = cae.train(features_df, epochs=5, batch_size=32, learning_rate=1e-3, verbose=False)
        print(f"✅ CAE trained successfully")
    except Exception as e:
        print(f"❌ CAE training failed: {e}")
        return False
    
    # Step 5: Test data preprocessing
    print("\n5. Testing data preprocessing...")
    try:
        preprocessor = DataPreprocessor(window_length=10, step_size=1)
        X_sequences, y_labels = preprocessor.prepare_data(features_df, stock_data['close'], cae, filtered_prices)
        print(f"✅ Data preprocessed: {X_sequences.shape}, {y_labels.shape}")
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return False
    
    # Step 6: Test PLSTM-TAL model
    print("\n6. Testing PLSTM-TAL model...")
    try:
        input_size = X_sequences.shape[2]
        model = PLSTM_TAL(input_size=input_size, hidden_size=32, num_layers=1, dropout=0.1)
        
        # Test forward pass
        import torch
        model.eval()
        with torch.no_grad():
            sample_input = torch.tensor(X_sequences[:5], dtype=torch.float32)
            logits, attention = model(sample_input)
            print(f"✅ PLSTM-TAL forward pass successful: {logits.shape}")
    except Exception as e:
        print(f"❌ PLSTM-TAL model failed: {e}")
        return False
    
    print("\n🎉 All integration tests passed!")
    print("The custom stock pipeline is ready for Streamlit integration.")
    return True

if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)