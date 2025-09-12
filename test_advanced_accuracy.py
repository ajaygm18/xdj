#!/usr/bin/env python3
"""
Advanced optimization script to achieve high accuracy (>75%) as requested.
Implements enhanced training strategies and feature engineering.
"""
import sys
import os
sys.path.append('./src')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from custom_stock_loader import CustomStockDataLoader
from indicators_talib import TechnicalIndicatorsTA
from train import DataPreprocessor
from model_plstm_tal import PLSTM_TAL
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """Advanced trainer with enhanced optimization techniques."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
    def train_with_advanced_optimization(self, X_train, y_train, X_val, y_val, 
                                       epochs=300, batch_size=16, 
                                       learning_rate=5e-4):
        """Train with advanced optimization techniques."""
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), 
                               lr=learning_rate, 
                               weight_decay=1e-4,
                               eps=1e-7)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15
        )
        
        # Focal loss for imbalanced classes
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, inputs, targets):
                bce_loss = nn.BCEWithLogitsLoss()(inputs, targets)
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
                return focal_loss.mean()
        
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        
        best_val_acc = 0
        patience = 50
        patience_counter = 0
        
        print(f"Starting advanced training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Create mini-batches
            num_batches = len(X_train) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                model_output = self.model(X_batch)
                if isinstance(model_output, tuple):
                    logits, _ = model_output
                else:
                    logits = model_output
                
                loss = criterion(logits.squeeze(), y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                train_correct += (predictions == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                model_output = self.model(X_val)
                if isinstance(model_output, tuple):
                    logits, _ = model_output
                else:
                    logits = model_output
                
                loss = criterion(logits.squeeze(), y_val)
                val_loss = loss.item()
                
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                val_correct = (predictions == y_val).sum().item()
                val_total = y_val.size(0)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_advanced.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss: {train_loss/num_batches:.4f}, "
                      f"Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model_advanced.pth'))
        print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        self.model.eval()
        
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(X_test)
            if isinstance(model_output, tuple):
                logits, _ = model_output
            else:
                logits = model_output
            
            predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()
        
        return accuracy

def advanced_feature_engineering(features_df, stock_data):
    """Apply advanced feature engineering techniques."""
    
    # 1. Technical indicator ratios (using correct column names)
    if 'RSI' in features_df.columns and 'BBANDS' in features_df.columns:
        features_df['rsi_bb_ratio'] = features_df['RSI'] / (features_df['BBANDS'] + 1e-8)
    
    if 'MACD' in features_df.columns and 'MACDsignal' in features_df.columns:
        features_df['macd_signal_ratio'] = features_df['MACD'] / (features_df['MACDsignal'] + 1e-8)
    elif 'MACD' in features_df.columns:
        # If MACDsignal not available, create a simple moving average
        macd_ma = features_df['MACD'].rolling(9).mean().fillna(features_df['MACD'])
        features_df['macd_signal_ratio'] = features_df['MACD'] / (macd_ma + 1e-8)
    
    # 2. Price momentum features
    prices = stock_data['close'].values
    for window in [5, 10, 20]:
        momentum = pd.Series(prices).pct_change(window).fillna(0)
        features_df[f'momentum_{window}'] = momentum
    
    # 3. Volatility features
    returns = pd.Series(prices).pct_change().fillna(0)
    for window in [5, 10, 20]:
        vol = returns.rolling(window).std().fillna(0)
        features_df[f'volatility_{window}'] = vol
    
    # 4. Volume-price features
    if 'volume' in stock_data.columns:
        volume = stock_data['volume'].values
        price_volume = prices * volume
        features_df['price_volume'] = price_volume / (price_volume.mean() + 1e-8)
    
    # 5. Lag features for key indicators
    key_indicators = ['RSI', 'MACD', 'CCI'] if all(col in features_df.columns for col in ['RSI', 'MACD', 'CCI']) else []
    
    # Add close price lags
    close_series = pd.Series(prices)
    for lag in [1, 2, 3]:
        features_df[f'close_lag_{lag}'] = close_series.shift(lag).fillna(close_series.iloc[0])
    
    # Add indicator lags
    for col in key_indicators:
        series = features_df[col]
        for lag in [1, 2]:
            features_df[f'{col}_lag_{lag}'] = series.shift(lag).fillna(series.iloc[0])
    
    # 6. Cross-indicator features
    if 'RSI' in features_df.columns and 'CCI' in features_df.columns:
        features_df['rsi_cci_product'] = features_df['RSI'] * features_df['CCI']
    
    if 'SMA' in features_df.columns and 'EMA' in features_df.columns:
        features_df['sma_ema_diff'] = features_df['SMA'] - features_df['EMA']
    
    # Fill any remaining NaN values
    features_df = features_df.fillna(features_df.mean())
    
    # Replace any infinite values
    features_df = features_df.replace([np.inf, -np.inf], 0)
    
    return features_df

def test_advanced_accuracy():
    """Test advanced optimization techniques for high accuracy."""
    print("=" * 70)
    print("ADVANCED ACCURACY OPTIMIZATION - TARGET: >75%")
    print("=" * 70)
    
    # Load 5 years of data
    print("\n1. Loading 5 years of AAPL data...")
    loader = CustomStockDataLoader()
    stock_data = loader.download_stock_data("AAPL", "2020-01-01", "2025-01-01")
    print(f"✅ Loaded {len(stock_data)} trading days")
    
    # Generate enhanced technical indicators
    print("\n2. Computing enhanced technical indicators...")
    try:
        indicators = TechnicalIndicatorsTA()
        features_df = indicators.compute_features(stock_data)
        print(f"✅ Generated {features_df.shape[1]} base TA-Lib features")
        
        # Apply advanced feature engineering
        features_df = advanced_feature_engineering(features_df, stock_data)
        print(f"✅ Enhanced to {features_df.shape[1]} total features")
        
    except ImportError:
        print("❌ TA-Lib not available, skipping test")
        return
    
    # Advanced data preparation
    print("\n3. Advanced data preparation...")
    preprocessor = DataPreprocessor(window_length=30, step_size=1)  # Longer window
    labels = preprocessor.create_labels(stock_data['close'])
    
    # Enhanced scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.values)
    
    min_length = min(len(features_scaled), len(labels))
    X_sequences, y_labels = preprocessor.create_sequences(
        features_scaled[:min_length], labels.values[:min_length]
    )
    
    print(f"✅ Created {len(X_sequences)} sequences of length 30")
    print(f"   Label distribution: {np.bincount(y_labels)}")
    
    # Stratified split with more training data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequences, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Enhanced PLSTM-TAL model
    print("\n4. Training enhanced PLSTM-TAL with advanced optimization...")
    
    enhanced_model = PLSTM_TAL(
        input_size=X_train.shape[2],
        hidden_size=128,  # Increased capacity
        num_layers=2,     # Deeper network
        dropout=0.2,      # Slightly higher dropout
        activation='tanh'
    )
    
    trainer = AdvancedModelTrainer(enhanced_model)
    
    # Train with advanced techniques
    best_val_acc = trainer.train_with_advanced_optimization(
        X_train, y_train, X_val, y_val,
        epochs=300,       # More epochs
        batch_size=16,    # Smaller batches
        learning_rate=5e-4  # Lower learning rate
    )
    
    # Final evaluation
    test_accuracy = trainer.evaluate(X_test, y_test)
    
    print(f"\n✅ Enhanced PLSTM-TAL Test Accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("ADVANCED OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Dataset: {len(stock_data)} days of AAPL data (5 years)")
    print(f"Features: {features_df.shape[1]} enhanced features")
    print(f"Window length: 30 (extended)")
    print(f"Model capacity: 128 hidden units, 2 layers")
    print(f"Training epochs: 300+ with advanced optimization")
    print("")
    print("FINAL RESULTS:")
    print(f"  Enhanced PLSTM-TAL: {test_accuracy*100:.1f}% accuracy")
    print(f"  Best validation:    {best_val_acc*100:.1f}% accuracy")
    print("")
    
    # Success criteria
    target_accuracy = 0.75
    if test_accuracy > target_accuracy:
        print("🎉 SUCCESS: High accuracy target achieved!")
        print("✅ Model performance significantly improved!")
        print("")
        print("ADVANCED OPTIMIZATIONS APPLIED:")
        print("  ✅ 1. Enhanced feature engineering (ratios, momentum, volatility)")
        print("  ✅ 2. Longer sequence windows (20 → 30)")
        print("  ✅ 3. Increased model capacity (64 → 128 hidden, 1 → 2 layers)")
        print("  ✅ 4. Advanced optimizer (AdamW with weight decay)")
        print("  ✅ 5. Focal loss for class imbalance")
        print("  ✅ 6. Learning rate scheduling")
        print("  ✅ 7. Enhanced gradient clipping")
        print("  ✅ 8. Extended training (200 → 300 epochs)")
    else:
        print("⚠️  Target accuracy not reached. Consider further optimization.")
        print(f"   Current: {test_accuracy*100:.1f}%")
        print(f"   Target: {target_accuracy*100:.1f}%+")
    
    print("\n" + "=" * 70)
    
    return test_accuracy

if __name__ == "__main__":
    test_advanced_accuracy()