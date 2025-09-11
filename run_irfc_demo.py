#!/usr/bin/env python3
"""
Script to run IRFC.NS demo through the Streamlit app and save results.
This script will programmatically run the full pipeline for IRFC.NS.
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import our modules
from src.custom_stock_loader import CustomStockDataLoader
from src.model_plstm_tal import PLSTM_TAL
from src.baselines import BaselineModelFactory
from src.cae import CAEFeatureExtractor
from src.train import DataPreprocessor, ModelTrainer
from src.eval import ModelEvaluator
from src.eemd import EEMDDenoiser

# Import indicators (try TA-Lib first, fallback to manual)
try:
    from src.indicators_talib import TechnicalIndicatorsTA
    # Test if TA-Lib actually works
    TechnicalIndicatorsTA()
    TALIB_AVAILABLE = True
    print("✅ TA-Lib available - using paper-compliant indicators")
except (ImportError, Exception):
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not available - using manual indicators")

from src.indicators import TechnicalIndicators

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_full_irfc_pipeline():
    """Run the complete IRFC.NS pipeline and save results."""
    
    print("🚀 Starting IRFC.NS Full Pipeline Demo")
    print("=" * 50)
    
    # Configuration
    stock_symbol = "IRFC.NS"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    
    results = {
        'stock_symbol': stock_symbol,
        'start_date': start_date,
        'end_date': end_date,
        'pipeline_steps': {},
        'accuracy_metrics': {},
        'model_paths': []
    }
    
    try:
        # Step 1: Load Stock Data
        print("\n📊 Step 1: Loading IRFC.NS stock data...")
        loader = CustomStockDataLoader()
        stock_data = loader.download_stock_data(stock_symbol, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            raise ValueError("Failed to load stock data")
            
        print(f"✅ Loaded {len(stock_data)} days of data")
        print(f"   Price range: ₹{stock_data['close'].min():.2f} - ₹{stock_data['close'].max():.2f}")
        
        results['pipeline_steps']['data_loading'] = {
            'status': 'success',
            'data_points': len(stock_data),
            'price_range': f"₹{stock_data['close'].min():.2f} - ₹{stock_data['close'].max():.2f}"
        }
        
        # Step 2: Technical Indicators
        print("\n🔧 Step 2: Computing technical indicators...")
        if TALIB_AVAILABLE:
            indicators = TechnicalIndicatorsTA()
            indicator_features = indicators.compute_features(stock_data)
            # Combine original data with indicators (using common indices)
            enhanced_data = pd.concat([stock_data, indicator_features], axis=1)
        else:
            indicators = TechnicalIndicators()
            enhanced_data = indicators.add_all_indicators(stock_data)
            
        print(f"✅ Added {indicator_features.shape[1] if TALIB_AVAILABLE else enhanced_data.shape[1] - stock_data.shape[1]} technical indicators")
        
        results['pipeline_steps']['technical_indicators'] = {
            'status': 'success',
            'indicator_count': indicator_features.shape[1] if TALIB_AVAILABLE else enhanced_data.shape[1] - stock_data.shape[1],
            'talib_used': TALIB_AVAILABLE
        }
        
        # Step 3: EEMD Denoising
        print("\n🔄 Step 3: EEMD denoising...")
        eemd = EEMDDenoiser()
        
        # Apply EEMD to the close price column for denoising
        close_prices = enhanced_data['close'].values
        denoised_close, eemd_info = eemd.denoise(close_prices)
        
        # Create denoised dataset by replacing close price with denoised version
        denoised_data = enhanced_data.copy()
        denoised_data['close'] = denoised_close
        
        print("✅ EEMD denoising completed")
        
        results['pipeline_steps']['eemd_denoising'] = {
            'status': 'success'
        }
        
        # Step 4: CAE Feature Extraction
        print("\n🧠 Step 4: CAE feature extraction...")
        cae = CAEFeatureExtractor(hidden_dim=128, encoding_dim=32)
        
        # Prepare data for CAE - ensure we have a clean numpy array
        feature_cols = [col for col in denoised_data.columns if col not in ['target']]
        X_features_df = denoised_data[feature_cols].copy()
        
        # Handle NaN values
        X_features_df = X_features_df.fillna(X_features_df.mean())
        
        print(f"   Prepared feature matrix: {X_features_df.shape}")
        
        # Train CAE (expects DataFrame)
        cae.train(X_features_df, epochs=30, batch_size=32)
        extracted_features = cae.extract_features(X_features_df)
        print(f"✅ CAE feature extraction completed - reduced to {extracted_features.shape[1]} features")
        
        results['pipeline_steps']['cae_extraction'] = {
            'status': 'success',
            'original_features': X_features_df.shape[1],
            'extracted_features': extracted_features.shape[1]
        }
        
        # Step 5: Prepare Training Data
        print("\n📈 Step 5: Preparing training data...")
        preprocessor = DataPreprocessor()
        
        # Create target variable (next day return)
        returns = denoised_data['close'].pct_change().shift(-1).fillna(0)
        target = (returns > 0).astype(int).values
        
        # Combine CAE features with target
        processed_data = np.column_stack([extracted_features, target])
        
        # Split data
        split_idx = int(0.8 * len(processed_data))
        train_data = processed_data[:split_idx]
        test_data = processed_data[split_idx:]
        
        print(f"✅ Training data: {len(train_data)} samples")
        print(f"   Testing data: {len(test_data)} samples")
        
        results['pipeline_steps']['data_preparation'] = {
            'status': 'success',
            'train_samples': len(train_data),
            'test_samples': len(test_data)
        }
        
        # Step 6: Train Models
        print("\n🔮 Step 6: Training PLSTM-TAL model...")
        
        # Prepare sequences for LSTM
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        # Create sequences
        def create_sequences(X, y, seq_length=60):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.LongTensor(y_train_seq)
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.LongTensor(y_test_seq)
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        model = PLSTM_TAL(input_size=input_size, hidden_size=64, num_layers=2)
        
        # Train model manually
        device = torch.device('cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        training_losses = []
        
        print("Training PLSTM-TAL model...")
        for epoch in range(50):
            epoch_loss = 0
            num_batches = 0
            
            # Create batches
            batch_size = 16  # Smaller batch size to avoid batch norm issues
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size].float()
                
                # Skip batches that are too small for batch norm
                if len(batch_X) < 2:
                    continue
                
                optimizer.zero_grad()
                
                logits, _ = model(batch_X)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                training_losses.append(avg_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/50: Loss = {avg_loss:.6f}")
        
        trained_model = model
        training_history = {'loss': training_losses}
        print("✅ PLSTM-TAL model training completed")
        
        # Step 7: Evaluate Model
        print("\n📊 Step 7: Evaluating model performance...")
        evaluator = ModelEvaluator()
        
        # Make predictions
        trained_model.eval()
        with torch.no_grad():
            train_logits, _ = trained_model(X_train_tensor)
            test_logits, _ = trained_model(X_test_tensor)
            
        # Convert logits to predictions (using sigmoid for binary classification)
        train_pred_probs = torch.sigmoid(train_logits).numpy()
        test_pred_probs = torch.sigmoid(test_logits).numpy()
        train_pred_np = (train_pred_probs > 0.5).astype(int)
        test_pred_np = (test_pred_probs > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        train_accuracy = accuracy_score(y_train_seq, train_pred_np)
        test_accuracy = accuracy_score(y_test_seq, test_pred_np)
        test_precision = precision_score(y_test_seq, test_pred_np, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_seq, test_pred_np, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_seq, test_pred_np, average='weighted', zero_division=0)
        
        print(f"✅ Model Performance:")
        print(f"   Training Accuracy: {train_accuracy:.4f}")
        print(f"   Testing Accuracy: {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1-Score: {test_f1:.4f}")
        
        results['accuracy_metrics'] = {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(test_f1)
        }
        
        results['pipeline_steps']['model_training'] = {
            'status': 'success',
            'model_type': 'PLSTM-TAL',
            'epochs': 50
        }
        
        # Step 8: Save Model
        print("\n💾 Step 8: Saving model...")
        model_path = f"plstm_tal_{stock_symbol.replace('.', '_')}_demo.pth"
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_config': {
                'input_size': input_size,
                'hidden_size': 64,
                'num_layers': 2
            },
            'training_history': training_history,
            'test_accuracy': test_accuracy,
            'test_predictions': test_pred_np.tolist(),
            'test_probabilities': test_pred_probs.tolist()
        }, model_path)
        
        print(f"✅ Model saved to: {model_path}")
        results['model_paths'].append(model_path)
        
        # Save results
        results_path = f"irfc_demo_results_{int(time.time())}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_path}")
        
        # Generate summary plot
        print("\n📈 Generating performance plot...")
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Stock price
        plt.subplot(2, 2, 1)
        plt.plot(stock_data['close'])
        plt.title(f'{stock_symbol} Stock Price')
        plt.ylabel('Price (₹)')
        
        # Plot 2: Training loss
        if 'loss' in training_history:
            plt.subplot(2, 2, 2)
            plt.plot(training_history['loss'])
            plt.title('Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
        
        # Plot 3: Predictions vs Actual
        plt.subplot(2, 2, 3)
        plt.scatter(y_test_seq, test_pred_np, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.title('Predictions vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # Plot 4: Performance metrics
        plt.subplot(2, 2, 4)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [test_accuracy, test_precision, test_recall, test_f1]
        plt.bar(metrics, values)
        plt.title('Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plot_path = f"irfc_demo_performance_{int(time.time())}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance plot saved to: {plot_path}")
        
        print("\n🎉 IRFC.NS Pipeline Demo Completed Successfully!")
        print("=" * 50)
        print(f"📊 Final Test Accuracy: {test_accuracy:.4f}")
        print(f"💾 Model saved: {model_path}")
        print(f"📄 Results saved: {results_path}")
        print(f"📈 Plot saved: {plot_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        results['pipeline_steps']['error'] = {
            'status': 'failed',
            'error': str(e)
        }
        return results

if __name__ == "__main__":
    print("🚀 IRFC.NS Demo - Running Full Pipeline")
    results = run_full_irfc_pipeline()
    print("\n✅ Demo completed!")