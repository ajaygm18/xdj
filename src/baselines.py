"""
Baseline models for comparison: CNN, LSTM, SVM, and Random Forest.
Implements simple versions of baseline models as specified in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class SimpleCNN(nn.Module):
    """Simple CNN baseline for sequence classification."""
    
    def __init__(self, input_size: int, seq_len: int, num_filters: int = 64, 
                 filter_sizes: list = [3, 4, 5], dropout: float = 0.1):
        """
        Initialize Simple CNN.
        
        Args:
            input_size: Input feature dimension
            seq_len: Sequence length
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
            dropout: Dropout rate
        """
        super(SimpleCNN, self).__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, num_filters, filter_size)
            for filter_size in filter_sizes
        ])
        
        # Dropout and classification
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Logits tensor (batch_size,)
        """
        # Transpose for Conv1d: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, num_filters, conv_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Apply dropout and classify
        x = self.dropout(x)
        logits = self.classifier(x).squeeze(-1)  # (batch_size,)
        
        return logits


class SimpleLSTM(nn.Module):
    """Simple LSTM baseline (no peepholes, no attention)."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.1, bidirectional: bool = False):
        """
        Initialize Simple LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(SimpleLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Standard LSTM (no peepholes)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Classification head
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Logits tensor (batch_size,)
        """
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 2*hidden_size)
        else:
            final_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        # Classify
        logits = self.classifier(final_hidden).squeeze(-1)  # (batch_size,)
        
        return logits


class SVMClassifier:
    """Support Vector Machine baseline classifier."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        Fit SVM model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Fitted SVM classifier
        """
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict_proba(X_scaled)[:, 1]  # Return positive class probability


class RandomForestClassifier_:
    """Random Forest baseline classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 random_state: int = 42):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier_':
        """
        Fit Random Forest model.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Fitted Random Forest classifier
        """
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Flatten sequences if needed
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, seq_len * n_features)
        
        return self.model.predict_proba(X)[:, 1]  # Return positive class probability


class BaselineModelFactory:
    """Factory for creating baseline models."""
    
    @staticmethod
    def create_model(model_type: str, input_size: int, seq_len: int = None, **kwargs) -> Any:
        """
        Create baseline model.
        
        Args:
            model_type: Type of model ('cnn', 'lstm', 'svm', 'rf')
            input_size: Input feature dimension
            seq_len: Sequence length (for neural models)
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model
        """
        model_type = model_type.lower()
        
        if model_type == 'cnn':
            if seq_len is None:
                raise ValueError("seq_len required for CNN model")
            return SimpleCNN(input_size=input_size, seq_len=seq_len, **kwargs)
        
        elif model_type == 'lstm':
            return SimpleLSTM(input_size=input_size, **kwargs)
        
        elif model_type == 'svm':
            return SVMClassifier(**kwargs)
        
        elif model_type == 'rf' or model_type == 'random_forest':
            return RandomForestClassifier_(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test baseline models
    print("Testing baseline models...")
    
    # Test data
    batch_size = 32
    seq_len = 20
    input_size = 48
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    X_torch = torch.randn(batch_size, seq_len, input_size)
    X_numpy = X_torch.numpy()
    y_numpy = np.random.randint(0, 2, batch_size)
    
    print(f"Test data shapes: X={X_torch.shape}, y={y_numpy.shape}")
    
    # Test neural network models
    print("\n--- Neural Network Models ---")
    
    # Test CNN
    cnn_model = BaselineModelFactory.create_model('cnn', input_size, seq_len)
    cnn_model.eval()
    with torch.no_grad():
        cnn_output = cnn_model(X_torch)
    print(f"CNN output shape: {cnn_output.shape}")
    
    # Test LSTM
    lstm_model = BaselineModelFactory.create_model('lstm', input_size)
    lstm_model.eval()
    with torch.no_grad():
        lstm_output = lstm_model(X_torch)
    print(f"LSTM output shape: {lstm_output.shape}")
    
    # Test traditional ML models
    print("\n--- Traditional ML Models ---")
    
    # Test SVM
    svm_model = BaselineModelFactory.create_model('svm', input_size)
    svm_model.fit(X_numpy, y_numpy)
    svm_pred = svm_model.predict(X_numpy)
    svm_proba = svm_model.predict_proba(X_numpy)
    print(f"SVM predictions shape: {svm_pred.shape}, probabilities shape: {svm_proba.shape}")
    
    # Test Random Forest
    rf_model = BaselineModelFactory.create_model('rf', input_size)
    rf_model.fit(X_numpy, y_numpy)
    rf_pred = rf_model.predict(X_numpy)
    rf_proba = rf_model.predict_proba(X_numpy)
    print(f"RF predictions shape: {rf_pred.shape}, probabilities shape: {rf_proba.shape}")
    
    print("\nBaseline models testing completed successfully!")