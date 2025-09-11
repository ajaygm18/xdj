"""
Quick test script to verify that the Streamlit app can complete a full model training cycle.
This runs a simplified version to test the complete pipeline.
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
from src.baselines import BaselineModelFactory
import torch

def test_quick_training_pipeline():
    """Test a quick training pipeline to verify everything works."""
    print("🧪 Testing Quick Training Pipeline")
    print("=" * 50)
    
    # Step 1: Load stock data (small dataset)
    print("\n1. Loading stock data (1 year)...")
    loader = CustomStockDataLoader()
    stock_data = loader.download_stock_data("AAPL", "2021-01-01", "2022-01-01")
    print(f"✅ Loaded {len(stock_data)} days of AAPL data")
    
    # Step 2: Generate technical indicators
    print("\n2. Computing technical indicators...")
    indicators = TechnicalIndicators()
    features_df = indicators.compute_features(stock_data)
    print(f"✅ Generated {len(features_df.columns)} indicators")
    
    # Step 3: Quick EEMD (simplified)
    print("\n3. Applying quick EEMD...")
    denoiser = EEMDDenoiser(n_ensembles=5, noise_scale=0.2, w=5)  # Very fast settings
    filtered_prices, metadata = denoiser.process_price_series(stock_data['close'])
    print(f"✅ EEMD completed")
    
    # Step 4: Quick CAE training
    print("\n4. Training CAE (quick)...")
    cae = CAEFeatureExtractor(hidden_dim=16, encoding_dim=4, dropout=0.1, lambda_reg=1e-4)
    cae_history = cae.train(features_df, epochs=3, batch_size=32, learning_rate=1e-3, verbose=False)
    print(f"✅ CAE trained")
    
    # Step 5: Data preprocessing
    print("\n5. Preprocessing data...")
    preprocessor = DataPreprocessor(window_length=5, step_size=1)  # Shorter sequences
    X_sequences, y_labels = preprocessor.prepare_data(features_df, stock_data['close'], cae, filtered_prices)
    print(f"✅ Data shape: {X_sequences.shape}, {y_labels.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    input_size = X_train.shape[2]
    print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 6: Quick model training
    print("\n6. Training PLSTM-TAL (quick)...")
    plstm_model = PLSTM_TAL(
        input_size=input_size,
        hidden_size=16,  # Smaller model
        num_layers=1,
        dropout=0.1,
        activation='tanh'
    )
    
    trainer = ModelTrainer(plstm_model)
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=3,  # Very quick training
        batch_size=16,
        learning_rate=1e-3,
        optimizer_name='adamax',
        early_stopping_patience=10
    )
    print(f"✅ Model trained for {len(history['train_loss'])} epochs")
    
    # Step 7: Quick evaluation
    print("\n7. Evaluating model...")
    evaluator = ModelEvaluator(save_plots=False)
    result = evaluator.evaluate_model(plstm_model, X_test, y_test, "PLSTM-TAL", is_torch_model=True)
    
    print(f"✅ Model evaluation completed:")
    print(f"   Accuracy: {result['metrics']['accuracy']:.3f}")
    print(f"   Precision: {result['metrics']['precision']:.3f}")
    print(f"   Recall: {result['metrics']['recall']:.3f}")
    print(f"   F1 Score: {result['metrics']['f1_score']:.3f}")
    
    # Step 8: Test baseline for comparison
    print("\n8. Training quick baseline...")
    svm_model = BaselineModelFactory.create_model('svm', input_size, C=1.0, gamma='scale')
    svm_model.fit(X_train, y_train)
    svm_result = evaluator.evaluate_model(svm_model, X_test, y_test, "SVM", is_torch_model=False)
    print(f"✅ SVM Accuracy: {svm_result['metrics']['accuracy']:.3f}")
    
    # Summary
    print(f"\n🎉 Pipeline test completed successfully!")
    print(f"PLSTM-TAL vs SVM: {result['metrics']['accuracy']:.3f} vs {svm_result['metrics']['accuracy']:.3f}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_quick_training_pipeline()
        print("\n✅ All tests passed! The Streamlit app should work correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)