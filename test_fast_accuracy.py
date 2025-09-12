#!/usr/bin/env python3
"""
Fast demo script to achieve high model accuracy with streamlined training.
Optimized for quick demonstration while maintaining high performance.
"""
import sys
import os
sys.path.append('./src')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from custom_stock_loader import CustomStockDataLoader
from indicators import TechnicalIndicators
from model_plstm_tal import PLSTM_TAL
from train import DataPreprocessor, ModelTrainer
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib for better indicators
try:
    from indicators_talib import TechnicalIndicatorsTA
    TALIB_AVAILABLE = True
    print("✅ TA-Lib available - using professional indicators")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️  TA-Lib not available - using fallback indicators")

def disable_timeouts():
    """Disable various timeout mechanisms."""
    import signal
    signal.alarm(0)
    os.environ['STREAMLIT_SERVER_ENABLE_WATCHDOG'] = 'false'
    os.environ['STREAMLIT_SERVER_WATCHDOG_TIMEOUT'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    print("✅ Workspace timeouts disabled")

def fast_high_accuracy_demo():
    """Fast demo to achieve high model accuracy."""
    print("=" * 80)
    print("🚀 FAST HIGH ACCURACY DEMO")
    print("=" * 80)
    print("Target: Achieve 75%+ accuracy with optimized training")
    print("Data: 5+ years of AAPL data")
    print("Model: Optimized PLSTM-TAL")
    
    # Disable timeouts as requested
    disable_timeouts()
    
    # Load 5+ years of data as requested
    print(f"\n📊 Loading 5+ years of AAPL data...")
    from datetime import datetime, timedelta
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365 + 100)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    loader = CustomStockDataLoader()
    stock_data = loader.download_stock_data("AAPL", start_str, end_str)
    
    print(f"✅ Loaded {len(stock_data)} trading days ({start_str} to {end_str})")
    
    # Compute enhanced features
    print("\n🔧 Computing technical indicators...")
    if TALIB_AVAILABLE:
        try:
            indicators = TechnicalIndicatorsTA()
            features_df = indicators.compute_features(stock_data)
            print(f"✅ Generated {features_df.shape[1]} TA-Lib indicators")
        except Exception as e:
            print(f"⚠️  TA-Lib failed: {e}, using fallback")
            indicators = TechnicalIndicators()
            features_df = indicators.compute_features(stock_data)
            print(f"✅ Generated {features_df.shape[1]} manual indicators")
    else:
        indicators = TechnicalIndicators()
        features_df = indicators.compute_features(stock_data)
        print(f"✅ Generated {features_df.shape[1]} manual indicators")
    
    # Advanced feature engineering
    print("\n🔄 Advanced feature engineering...")
    prices = stock_data['close']
    
    # Add momentum features
    for window in [5, 10, 20]:
        momentum = prices.pct_change(window).fillna(0)
        features_df[f'momentum_{window}'] = momentum
    
    # Add volatility features
    returns = prices.pct_change().fillna(0)
    for window in [5, 10, 20]:
        vol = returns.rolling(window).std().fillna(0)
        features_df[f'volatility_{window}'] = vol
    
    # Add volume features
    volumes = stock_data['volume']
    for window in [5, 10]:
        vol_ma = volumes.rolling(window).mean()
        vol_ratio = volumes / vol_ma
        features_df[f'volume_ratio_{window}'] = vol_ratio.fillna(1)
    
    print(f"✅ Enhanced to {features_df.shape[1]} features")
    
    # Data preprocessing
    preprocessor = DataPreprocessor(window_length=20, step_size=1)
    labels = preprocessor.create_labels(stock_data['close'])
    
    # Scale features
    features_scaled = preprocessor.feature_scaler.fit_transform(features_df.values)
    
    # Align data
    min_length = min(len(features_scaled), len(labels))
    features_scaled = features_scaled[:min_length]
    labels_array = labels.values[:min_length]
    
    # Create sequences
    X_sequences, y_labels = preprocessor.create_sequences(features_scaled, labels_array)
    
    print(f"✅ Created {len(X_sequences)} sequences")
    print(f"   Label distribution: {np.bincount(y_labels)}")
    
    # Train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Create optimized model
    print("\n🧠 Creating optimized PLSTM-TAL model...")
    
    model = PLSTM_TAL(
        input_size=X_train.shape[2],
        hidden_size=64,  # Optimized size
        num_layers=2,    # Balanced depth
        dropout=0.2,
        bidirectional=False,  # Faster training
        activation='tanh'
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Optimized training
    print("\n🏋️  Fast optimized training...")
    trainer = ModelTrainer(model)
    
    # Train with aggressive but stable parameters
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=100,           # Fewer epochs for speed
        batch_size=64,        # Larger batches for efficiency
        learning_rate=2e-3,   # Higher learning rate for faster convergence
        optimizer_name='adamw',  # Better optimizer
        early_stopping_patience=25
    )
    
    # Comprehensive evaluation
    print("\n📊 Model evaluation...")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        output = model(X_test_tensor)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        
        test_probs = torch.sigmoid(logits)
        test_preds = (test_probs > 0.5).float()
        test_accuracy = (test_preds == y_test_tensor).float().mean().item()
    
    # Validation evaluation
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        output = model(X_val_tensor)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        
        val_probs = torch.sigmoid(logits)
        val_preds = (val_probs > 0.5).float()
        val_accuracy = (val_preds == y_val_tensor).float().mean().item()
    
    # Training evaluation
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        output = model(X_train_tensor)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        
        train_probs = torch.sigmoid(logits)
        train_preds = (train_probs > 0.5).float()
        train_accuracy = (train_preds == y_train_tensor).float().mean().item()
    
    # Calculate additional metrics for test set
    test_preds_np = test_preds.numpy()
    y_test_np = y_test_tensor.numpy()
    
    # Precision, Recall, F1
    tp = ((test_preds_np == 1) & (y_test_np == 1)).sum()
    fp = ((test_preds_np == 1) & (y_test_np == 0)).sum()
    fn = ((test_preds_np == 0) & (y_test_np == 1)).sum()
    tn = ((test_preds_np == 0) & (y_test_np == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display results
    print("\n" + "=" * 80)
    print("🎯 FAST HIGH ACCURACY RESULTS")
    print("=" * 80)
    
    print(f"Dataset: {len(stock_data)} days of AAPL data (5+ years)")
    print(f"Features: {features_df.shape[1]} enhanced indicators")
    print(f"Sequences: {len(X_sequences)} training windows")
    print(f"Model: Optimized PLSTM-TAL ({total_params:,} parameters)")
    print("")
    
    print("PERFORMANCE METRICS:")
    print(f"  📈 Training Accuracy:   {train_accuracy*100:.2f}%")
    print(f"  📊 Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"  🎯 Test Accuracy:       {test_accuracy*100:.2f}%")
    print("")
    print(f"  🎯 Test Precision:      {precision:.3f}")
    print(f"  🎯 Test Recall:         {recall:.3f}")
    print(f"  🎯 Test F1 Score:       {f1:.3f}")
    print("")
    
    # Success assessment
    target_accuracy = 0.75  # 75%+ target
    achieved_accuracy = test_accuracy
    
    if achieved_accuracy >= target_accuracy:
        print("🎉 SUCCESS: HIGH ACCURACY ACHIEVED!")
        print(f"✅ Target: {target_accuracy*100:.1f}%+ | Achieved: {achieved_accuracy*100:.2f}%")
        status = "SUCCESS"
    elif achieved_accuracy >= 0.65:
        print("✅ GOOD PERFORMANCE: Significant improvement achieved")
        print(f"📊 Achieved: {achieved_accuracy*100:.2f}% (target: {target_accuracy*100:.1f}%+)")
        status = "GOOD"
    else:
        print("⚠️  NEEDS IMPROVEMENT: Additional optimization required")
        print(f"📊 Current: {achieved_accuracy*100:.2f}% (target: {target_accuracy*100:.1f}%+)")
        status = "NEEDS_IMPROVEMENT"
    
    # Save model
    torch.save(model.state_dict(), 'fast_high_accuracy_model.pth')
    print("✅ Model saved as 'fast_high_accuracy_model.pth'")
    
    print("\n" + "=" * 80)
    print("IMPROVEMENTS IMPLEMENTED:")
    print("✅ 1. Enhanced feature engineering (momentum, volatility, volume)")
    print("✅ 2. Optimized PLSTM-TAL architecture")
    print("✅ 3. Advanced training with AdamW optimizer")
    print("✅ 4. 5+ years of training data as requested")
    print("✅ 5. Workspace timeouts disabled as requested")
    print("✅ 6. Professional TA-Lib indicators when available")
    print("✅ 7. Comprehensive evaluation metrics")
    print("✅ 8. Fast training for quick demonstration")
    
    return {
        'model': model,
        'test_accuracy': achieved_accuracy,
        'val_accuracy': val_accuracy,
        'train_accuracy': train_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'status': status,
        'total_params': total_params,
        'test_predictions': test_preds_np,
        'test_probabilities': test_probs.numpy()
    }

if __name__ == "__main__":
    print("🚀 Starting Fast High Accuracy Demo...")
    print("This will achieve high accuracy using optimized training.")
    print("")
    
    results = fast_high_accuracy_demo()
    
    print(f"\n🏁 Demo completed!")
    print(f"Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"Status: {results['status']}")
    
    # Save results
    with open('fast_accuracy_results.json', 'w') as f:
        json.dump({
            'test_accuracy': results['test_accuracy'],
            'val_accuracy': results['val_accuracy'],
            'train_accuracy': results['train_accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'status': results['status'],
            'total_params': results['total_params']
        }, f, indent=2)
    
    print("📄 Results saved to 'fast_accuracy_results.json'")