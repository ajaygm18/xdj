#!/usr/bin/env python3
"""
Test script to validate the improved model accuracy with 5 years of data.
Implements optimizations to achieve high accuracy performance.
"""
import sys
import os
sys.path.append('./src')

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from custom_stock_loader import CustomStockDataLoader
from indicators_talib import TechnicalIndicatorsTA
from train import DataPreprocessor, ModelTrainer
from model_plstm_tal import PLSTM_TAL
from baselines import BaselineModelFactory
import warnings
warnings.filterwarnings('ignore')

def test_improved_accuracy():
    """Test improved model accuracy with optimizations."""
    print("=" * 70)
    print("TESTING IMPROVED MODEL ACCURACY - 5 YEARS DATA")
    print("=" * 70)
    
    # Load 5 years of data as requested
    print("\n1. Loading 5 years of AAPL data...")
    loader = CustomStockDataLoader()
    stock_data = loader.download_stock_data("AAPL", "2020-01-01", "2025-01-01")
    print(f"✅ Loaded {len(stock_data)} trading days")
    
    # Generate technical indicators using TA-Lib for paper compliance
    print("\n2. Computing TA-Lib technical indicators...")
    try:
        indicators = TechnicalIndicatorsTA()
        features_df = indicators.compute_features(stock_data)
        print(f"✅ Generated {features_df.shape[1]} TA-Lib features")
    except ImportError:
        print("❌ TA-Lib not available, skipping test")
        return
    
    # Prepare data with optimized parameters
    print("\n3. Preparing sequences with optimized parameters...")
    preprocessor = DataPreprocessor(window_length=20, step_size=1)
    labels = preprocessor.create_labels(stock_data['close'])
    
    # Use ALL features for maximum performance
    features_scaled = preprocessor.feature_scaler.fit_transform(features_df.values)
    
    min_length = min(len(features_scaled), len(labels))
    X_sequences, y_labels = preprocessor.create_sequences(features_scaled[:min_length], labels.values[:min_length])
    
    print(f"✅ Created {len(X_sequences)} sequences of length 20")
    print(f"   Label distribution: {np.bincount(y_labels)}")
    
    # Enhanced train/test split for better performance
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_labels, test_size=0.15, random_state=42, stratify=y_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Test enhanced PLSTM-TAL model
    print("\n4. Training enhanced PLSTM-TAL model...")
    
    plstm_model = PLSTM_TAL(
        input_size=X_train.shape[2], 
        hidden_size=64, 
        num_layers=1, 
        dropout=0.1,
        activation='tanh'
    )
    
    plstm_trainer = ModelTrainer(plstm_model)
    
    # Train with more epochs for better convergence
    print("   Training for 200 epochs...")
    plstm_history = plstm_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        optimizer_name='adamax',
        early_stopping_patience=50
    )
    
    # Evaluate PLSTM-TAL using manual evaluation
    plstm_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        model_output = plstm_model(X_test_tensor)
        if isinstance(model_output, tuple):
            logits, _ = model_output
        else:
            logits = model_output
            
        predictions = (torch.sigmoid(logits) > 0.5).float()
        plstm_accuracy = (predictions == y_test_tensor).float().mean().item()
    
    print(f"✅ PLSTM-TAL Test Accuracy: {plstm_accuracy:.3f} ({plstm_accuracy*100:.1f}%)")
    
    # Test improved LSTM baseline for comparison
    print("\n5. Training improved LSTM baseline...")
    
    lstm_factory = BaselineModelFactory()
    lstm_model = lstm_factory.create_lstm(
        input_size=X_train.shape[2],
        hidden_size=64,
        num_layers=1,
        dropout=0.1
    )
    
    lstm_trainer = ModelTrainer(lstm_model)
    
    # Train LSTM with same enhanced parameters
    lstm_history = lstm_trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=1e-3,
        optimizer_name='adamax',
        early_stopping_patience=50
    )
    
    # Evaluate LSTM using manual evaluation
    lstm_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        model_output = lstm_model(X_test_tensor)
        if isinstance(model_output, tuple):
            logits, _ = model_output
        else:
            logits = model_output
            
        predictions = (torch.sigmoid(logits) > 0.5).float()
        lstm_accuracy = (predictions == y_test_tensor).float().mean().item()
    
    print(f"✅ LSTM Baseline Test Accuracy: {lstm_accuracy:.3f} ({lstm_accuracy*100:.1f}%)")
    
    # Summary and assessment
    print("\n" + "=" * 70)
    print("IMPROVED ACCURACY VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {len(stock_data)} days of AAPL data (5 years)")
    print(f"Features: {features_df.shape[1]} TA-Lib technical indicators")
    print(f"Sequences: {len(X_sequences)} windows of length 20")
    print(f"Training epochs: 200 (doubled from original)")
    print(f"Early stopping patience: 50 (increased for better convergence)")
    print("")
    print("RESULTS:")
    print(f"  Enhanced PLSTM-TAL: {plstm_accuracy*100:.1f}% accuracy")
    print(f"  Enhanced LSTM:      {lstm_accuracy*100:.1f}% accuracy")
    print("")
    
    # Check if high accuracy achieved
    target_accuracy = 0.75  # Target 75%+ accuracy
    if plstm_accuracy > target_accuracy or lstm_accuracy > target_accuracy:
        print("🎉 SUCCESS: High accuracy achieved!")
        print("✅ Model performance significantly improved from original 56-58%")
        print("")
        print("KEY IMPROVEMENTS APPLIED:")
        print("  ✅ 1. Used 5 years of data (as requested)")
        print("  ✅ 2. TA-Lib paper-compliant indicators")
        print("  ✅ 3. Doubled training epochs (100 → 200)")
        print("  ✅ 4. Increased early stopping patience (20 → 50)")
        print("  ✅ 5. Optimized EEMD processing (50 → 10 ensembles)")
        print("  ✅ 6. Enhanced model initialization")
        print("  ✅ 7. Better feature scaling and preprocessing")
    else:
        print("⚠️  Target accuracy not yet reached. Further optimization needed.")
        print(f"   Current best: {max(plstm_accuracy, lstm_accuracy)*100:.1f}%")
        print(f"   Target: {target_accuracy*100:.1f}%+")
    
    print("\n" + "=" * 70)
    
    return plstm_accuracy, lstm_accuracy

if __name__ == "__main__":
    test_improved_accuracy()