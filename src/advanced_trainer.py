"""
Advanced training module with sophisticated optimization techniques for high accuracy.
Includes ensemble training, progressive learning, and advanced regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Tuple, Dict, List, Any, Optional
import os
import warnings
warnings.filterwarnings('ignore')


class AdvancedDataPreprocessor:
    """Advanced data preprocessing with multiple scaling strategies and feature engineering."""
    
    def __init__(self, window_length: int = 30, step_size: int = 1, 
                 scaling_method: str = 'robust'):
        """
        Initialize advanced data preprocessor.
        
        Args:
            window_length: Length of input sequences
            step_size: Step size for sliding window
            scaling_method: Scaling method ('robust', 'standard', 'minmax')
        """
        self.window_length = window_length
        self.step_size = step_size
        self.scaling_method = scaling_method
        
        if scaling_method == 'robust':
            self.feature_scaler = RobustScaler()
            self.price_scaler = RobustScaler()
        elif scaling_method == 'standard':
            self.feature_scaler = StandardScaler()
            self.price_scaler = StandardScaler()
        else:  # minmax
            from sklearn.preprocessing import MinMaxScaler
            self.feature_scaler = MinMaxScaler()
            self.price_scaler = MinMaxScaler()
    
    def create_advanced_labels(self, prices: pd.Series, method: str = 'multi_horizon') -> pd.Series:
        """
        Create advanced labels with multiple prediction horizons.
        
        Args:
            prices: Price series
            method: Label creation method
            
        Returns:
            Advanced labels series
        """
        if method == 'multi_horizon':
            # Predict if price will be higher in next 1-3 days (majority vote)
            future_returns = []
            for horizon in [1, 2, 3]:
                returns = (prices.shift(-horizon) / prices - 1) > 0
                future_returns.append(returns)
            
            # Majority vote across horizons
            labels = pd.DataFrame(future_returns).T.sum(axis=1) >= 2
            return labels.astype(int)[:-3]  # Remove last 3 values
            
        elif method == 'trend_following':
            # Predict trend continuation (price above/below moving average)
            ma_5 = prices.rolling(5).mean()
            ma_10 = prices.rolling(10).mean()
            
            future_price = prices.shift(-1)
            future_trend = ((future_price > ma_5.shift(-1)) & 
                          (ma_5.shift(-1) > ma_10.shift(-1))).astype(int)
            
            return future_trend[:-1]
            
        else:  # simple directional
            price_changes = prices.shift(-1) - prices
            labels = (price_changes > 0).astype(int)
            return labels[:-1]
    
    def add_feature_engineering(self, features_df: pd.DataFrame, 
                               stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced feature engineering.
        
        Args:
            features_df: Base features dataframe
            stock_data: Stock price data
            
        Returns:
            Enhanced features dataframe
        """
        enhanced_features = features_df.copy()
        prices = stock_data['close']
        volumes = stock_data['volume']
        highs = stock_data['high']
        lows = stock_data['low']
        
        # Price momentum features
        for window in [3, 5, 10, 20]:
            momentum = prices.pct_change(window).fillna(0)
            enhanced_features[f'momentum_{window}'] = momentum
            
            # Momentum acceleration
            momentum_change = momentum.diff().fillna(0)
            enhanced_features[f'momentum_accel_{window}'] = momentum_change
        
        # Volatility features
        returns = prices.pct_change().fillna(0)
        for window in [5, 10, 20]:
            vol = returns.rolling(window).std().fillna(0)
            enhanced_features[f'volatility_{window}'] = vol
            
            # GARCH-like volatility clustering
            vol_of_vol = vol.rolling(window).std().fillna(0)
            enhanced_features[f'vol_of_vol_{window}'] = vol_of_vol
        
        # Volume features
        for window in [5, 10, 20]:
            vol_ma = volumes.rolling(window).mean()
            vol_ratio = volumes / vol_ma
            enhanced_features[f'volume_ratio_{window}'] = vol_ratio.fillna(1)
            
            # Price-volume interaction
            pv_trend = (returns * vol_ratio).rolling(window).sum().fillna(0)
            enhanced_features[f'pv_trend_{window}'] = pv_trend
        
        # Volatility regime features
        high_vol_threshold = returns.rolling(60).std().quantile(0.8)
        enhanced_features['high_vol_regime'] = (
            returns.rolling(20).std() > high_vol_threshold
        ).astype(int)
        
        # Market microstructure features
        bid_ask_spread = (highs - lows) / prices
        enhanced_features['bid_ask_spread'] = bid_ask_spread.fillna(0)
        
        # Support/Resistance levels
        for window in [20, 50]:
            resistance = highs.rolling(window).max()
            support = lows.rolling(window).min()
            position = (prices - support) / (resistance - support)
            enhanced_features[f'sr_position_{window}'] = position.fillna(0.5)
        
        # Fractal features (simplified)
        for window in [5, 10]:
            local_max = prices.rolling(window, center=True).max() == prices
            local_min = prices.rolling(window, center=True).min() == prices
            enhanced_features[f'local_max_{window}'] = local_max.astype(int)
            enhanced_features[f'local_min_{window}'] = local_min.astype(int)
        
        return enhanced_features.fillna(method='ffill').fillna(0)
    
    def create_sequences_with_augmentation(self, features: np.ndarray, 
                                         labels: np.ndarray = None,
                                         augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with data augmentation.
        
        Args:
            features: Feature matrix
            labels: Labels array
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        n_samples, n_features = features.shape
        
        if n_samples < self.window_length:
            raise ValueError(f"Not enough samples ({n_samples}) for window length ({self.window_length})")
        
        sequences = []
        sequence_labels = []
        
        # Base sequences
        for i in range(0, n_samples - self.window_length + 1, self.step_size):
            seq = features[i:i + self.window_length]
            sequences.append(seq)
            
            if labels is not None:
                label_idx = i + self.window_length - 1
                if label_idx < len(labels):
                    sequence_labels.append(labels[label_idx])
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels) if sequence_labels else None
        
        # Data augmentation
        if augment and len(sequences) > 0:
            augmented_sequences = []
            augmented_labels = []
            
            for seq, label in zip(sequences, sequence_labels):
                # Original sequence
                augmented_sequences.append(seq)
                augmented_labels.append(label)
                
                # Add gaussian noise (small amount)
                noise_seq = seq + np.random.normal(0, 0.01, seq.shape)
                augmented_sequences.append(noise_seq)
                augmented_labels.append(label)
                
                # Time shifting (slight shifts)
                if np.random.random() > 0.7:
                    shift_amount = np.random.randint(-2, 3)
                    if shift_amount != 0:
                        shifted_seq = np.roll(seq, shift_amount, axis=0)
                        augmented_sequences.append(shifted_seq)
                        augmented_labels.append(label)
            
            sequences = np.array(augmented_sequences)
            sequence_labels = np.array(augmented_labels)
        
        return sequences, sequence_labels


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class AdvancedModelTrainer:
    """Advanced model trainer with ensemble methods and progressive learning."""
    
    def __init__(self, model: nn.Module, device: str = None):
        """
        Initialize advanced trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.training_history = []
        self.best_models = []
    
    def train_with_progressive_learning(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray, y_val: np.ndarray,
                                      config: dict) -> Dict[str, List]:
        """
        Train model with progressive learning strategy.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Training configuration
            
        Returns:
            Training history
        """
        training_config = config.get('training', {})
        epochs = training_config.get('epochs', 300)
        batch_size = training_config.get('batch_size', 64)
        learning_rate = training_config.get('learning_rate', 1e-4)
        warmup_epochs = training_config.get('warmup_epochs', 50)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Progressive learning stages
        stages = [
            {'epochs': warmup_epochs, 'lr': learning_rate * 0.1, 'loss': 'bce'},
            {'epochs': epochs - warmup_epochs, 'lr': learning_rate, 'loss': 'focal'}
        ]
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = training_config.get('patience', 100)
        
        for stage_idx, stage in enumerate(stages):
            print(f"\nStage {stage_idx + 1}: {stage['epochs']} epochs, LR: {stage['lr']}")
            
            # Setup optimizer and loss for this stage
            if training_config.get('optimizer', 'adamw').lower() == 'adamw':
                optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=stage['lr'],
                    weight_decay=training_config.get('weight_decay', 1e-5)
                )
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=stage['lr'])
            
            if stage['loss'] == 'focal':
                criterion = FocalLoss(alpha=1.0, gamma=2.0)
            else:
                criterion = nn.BCEWithLogitsLoss()
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage['epochs'], eta_min=stage['lr'] * 0.01
            )
            
            for epoch in range(stage['epochs']):
                # Training phase
                self.model.train()
                train_losses = []
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if hasattr(self.model, '__call__'):
                        output = self.model(batch_x)
                        if isinstance(output, tuple):
                            logits, _ = output
                        else:
                            logits = output
                    else:
                        logits = self.model(batch_x)
                    
                    loss = criterion(logits, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        training_config.get('gradient_clip', 1.0)
                    )
                    
                    optimizer.step()
                    train_losses.append(loss.item())
                
                scheduler.step()
                
                # Validation phase
                self.model.eval()
                val_losses = []
                correct_predictions = 0
                total_predictions = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        output = self.model(batch_x)
                        if isinstance(output, tuple):
                            logits, _ = output
                        else:
                            logits = output
                        
                        val_loss = F.binary_cross_entropy_with_logits(logits, batch_y)
                        val_losses.append(val_loss.item())
                        
                        # Calculate accuracy
                        predictions = (torch.sigmoid(logits) > 0.5).float()
                        correct_predictions += (predictions == batch_y).sum().item()
                        total_predictions += len(batch_y)
                
                # Record metrics
                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                val_accuracy = correct_predictions / total_predictions
                
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_enhanced_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Print progress
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_enhanced_model.pth'))
        
        return history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                output = self.model(batch_x)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = (all_predictions == all_labels).mean()
        
        # Precision, Recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Directional accuracy (for stock prediction)
        directional_accuracy = accuracy  # Same as accuracy for binary classification
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'directional_accuracy': directional_accuracy,
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }