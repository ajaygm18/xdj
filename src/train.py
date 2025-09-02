"""
Training module for all models (PLSTM-TAL and baselines).
Implements training loops, data preprocessing, and model management.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from typing import Tuple, Dict, List, Any, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# Import our modules (only the ones that work without TA-Lib)
from model_plstm_tal import PLSTM_TAL
from baselines import BaselineModelFactory
from cae import CAEFeatureExtractor
# import eemd
# import indicators
# import io as data_io


class DataPreprocessor:
    """Handles data preprocessing for model training."""
    
    def __init__(self, window_length: int = 20, step_size: int = 1):
        """
        Initialize data preprocessor.
        
        Args:
            window_length: Length of input sequences
            step_size: Step size for sliding window
        """
        self.window_length = window_length
        self.step_size = step_size
        self.feature_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        
    def create_labels(self, prices: pd.Series) -> pd.Series:
        """
        Create binary labels based on return movement direction (paper-compliant).
        
        Label rule from paper equation (1): y_i(t) = 1 if r_i(t+1) > r_i(t), else 0
        where r_i(t) is the return at time t.
        
        Args:
            prices: Price series
            
        Returns:
            Binary labels series
        """
        # Calculate returns: r(t) = log(price(t) / price(t-1))
        returns = np.log(prices / prices.shift(1)).fillna(0)
        
        # Create labels: 1 if return(t+1) > return(t), else 0
        # This matches paper equation (1) exactly
        labels = (returns.shift(-1) > returns).astype(int)
        
        # Drop last value (no future return available)
        labels = labels[:-1]
        
        return labels
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from features and labels.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Labels array (n_samples,)
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        n_samples, n_features = features.shape
        
        if n_samples < self.window_length:
            raise ValueError(f"Not enough samples ({n_samples}) for window length ({self.window_length})")
        
        sequences = []
        sequence_labels = []
        
        for i in range(0, n_samples - self.window_length + 1, self.step_size):
            # Extract sequence
            seq = features[i:i + self.window_length]
            sequences.append(seq)
            
            # Extract corresponding label (at the end of the sequence)
            if labels is not None:
                label_idx = i + self.window_length - 1
                if label_idx < len(labels):
                    sequence_labels.append(labels[label_idx])
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels) if sequence_labels else None
        
        return sequences, sequence_labels
    
    def prepare_data(self, features_df: pd.DataFrame, prices: pd.Series, 
                    cae_extractor: CAEFeatureExtractor = None, 
                    filtered_prices: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare complete dataset for training.
        
        Args:
            features_df: Technical indicators DataFrame
            prices: Price series for label creation
            cae_extractor: Trained CAE for feature extraction
            filtered_prices: EEMD filtered prices
            
        Returns:
            Tuple of (X_sequences, y_labels)
        """
        # Align data by index
        common_index = features_df.index.intersection(prices.index)
        features_aligned = features_df.loc[common_index]
        prices_aligned = prices.loc[common_index]
        
        # Scale technical indicators
        features_scaled = self.feature_scaler.fit_transform(features_aligned.values)
        
        # Extract CAE features if provided
        if cae_extractor is not None:
            cae_features = cae_extractor.extract_features(features_aligned)
            print(f"CAE features shape: {cae_features.shape}")
        else:
            cae_features = features_scaled
            print("Using raw features (no CAE)")
        
        # Add filtered price as additional feature if provided
        if filtered_prices is not None:
            # Align filtered prices
            filtered_aligned = filtered_prices.loc[common_index]
            
            # Scale filtered prices
            filtered_scaled = self.price_scaler.fit_transform(filtered_aligned.values.reshape(-1, 1)).flatten()
            
            # Combine CAE features with filtered price
            combined_features = np.column_stack([cae_features, filtered_scaled])
            print(f"Combined features shape (CAE + filtered price): {combined_features.shape}")
        else:
            combined_features = cae_features
            print("No filtered price added")
        
        # Create labels
        labels = self.create_labels(prices_aligned)
        
        # Align features and labels
        min_length = min(len(combined_features), len(labels))
        combined_features = combined_features[:min_length]
        labels = labels.values[:min_length]
        
        # Create sequences
        X_sequences, y_labels = self.create_sequences(combined_features, labels)
        
        print(f"Created {len(X_sequences)} sequences of length {self.window_length}")
        print(f"Feature dimension per timestep: {X_sequences.shape[2]}")
        print(f"Label distribution: {np.bincount(y_labels)}")
        
        return X_sequences, y_labels


class ModelTrainer:
    """Handles training of deep learning models."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            device: Device for training
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module) -> Dict[str, float]:
        """Train model for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            model_output = self.model(X_batch)
            if isinstance(model_output, tuple):
                # PLSTM-TAL returns (logits, attention_weights)
                logits, _ = model_output
            else:
                # Baseline models return only logits
                logits = model_output
            
            # Calculate loss
            loss = criterion(logits, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct_predictions += (predictions == y_batch).sum().item()
            total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float()
                
                # Forward pass
                model_output = self.model(X_batch)
                if isinstance(model_output, tuple):
                    # PLSTM-TAL returns (logits, attention_weights)
                    logits, _ = model_output
                else:
                    # Baseline models return only logits
                    logits = model_output
                
                # Calculate loss
                loss = criterion(logits, y_batch)
                
                # Statistics
                total_loss += loss.item()
                predictions = (torch.sigmoid(logits) > 0.5).float()
                correct_predictions += (predictions == y_batch).sum().item()
                total_samples += y_batch.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3,
              optimizer_name: str = 'adamax', early_stopping_patience: int = 10) -> Dict:
        """
        Train model with full training loop.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer_name: Optimizer type ('adamax', 'adam', 'sgd')
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), 
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer (Adamax as specified in paper)
        if optimizer_name.lower() == 'adamax':
            optimizer = optim.Adamax(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function (binary cross-entropy as specified in paper)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, criterion)
                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                
                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.4f}")
        
        # Restore best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored best model weights")
        
        print("Training completed!")
        return history
    
    def save_model(self, filepath: str, metadata: Dict = None) -> None:
        """Save trained model."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    # Test training pipeline
    print("Testing training pipeline...")
    
    # Create synthetic data for testing
    n_samples = 1000
    n_features = 40
    window_length = 20
    
    # Generate synthetic features and prices
    np.random.seed(42)
    features_data = np.random.randn(n_samples, n_features)
    features_df = pd.DataFrame(features_data, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Generate synthetic price series
    price_changes = np.random.normal(0.001, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(price_changes))
    prices_series = pd.Series(prices, index=features_df.index)
    
    print(f"Synthetic data: {features_df.shape} features, {len(prices_series)} prices")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(window_length=window_length)
    
    # Prepare data
    X_sequences, y_labels = preprocessor.prepare_data(features_df, prices_series)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"Training data: {X_train.shape}, {y_train.shape}")
    print(f"Validation data: {X_val.shape}, {y_val.shape}")
    
    # Test PLSTM-TAL training
    input_size = X_train.shape[2]
    model = PLSTM_TAL(
        input_size=input_size,
        hidden_size=32,  # Smaller for testing
        dropout=0.1,
        activation='tanh'
    )
    
    trainer = ModelTrainer(model)
    
    # Train for few epochs
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=5, batch_size=16, learning_rate=1e-3,
        optimizer_name='adamax'
    )
    
    print(f"Training history keys: {list(history.keys())}")
    print(f"Final training accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    print("Training pipeline testing completed successfully!")