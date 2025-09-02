"""
Contractive Autoencoder (CAE) implementation for feature extraction.
Implements the CAE with contractive penalty as described in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ContractiveAutoencoder(nn.Module):
    """
    Contractive Autoencoder with penalty term.
    
    Loss function: L_CAE = ∑L(x,g(h(X))) + λ∥J_h(X)∥_F^2
    where J_h(X) is the Jacobian of the encoder hidden layer activations.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, encoding_dim: int, dropout: float = 0.1):
        """
        Initialize CAE.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            encoding_dim: Encoding (bottleneck) dimension
            dropout: Dropout rate
        """
        super(ContractiveAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, encoding_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Store encoder hidden layer for Jacobian computation
        self.encoder_hidden = None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through CAE."""
        # Enable gradient computation for input
        x.requires_grad_(True)
        
        # Get hidden representation (before final encoding layer)
        h1 = self.encoder[0](x)  # Linear layer
        h2 = self.encoder[1](h1)  # ReLU
        h3 = self.encoder[2](h2)  # Dropout
        encoded = self.encoder[3](h3)  # Final encoding
        
        # Store hidden activations for Jacobian computation
        self.encoder_hidden = h2
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return encoded, reconstructed
    
    def contractive_loss(self, x: torch.Tensor, x_hat: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
        """
        Compute contractive loss with Jacobian penalty.
        
        Args:
            x: Input data
            x_hat: Reconstructed data
            lambda_reg: Regularization parameter for contractive penalty
            
        Returns:
            Total loss (reconstruction + contractive penalty)
        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(x_hat, x)
        
        # Contractive penalty: Frobenius norm of Jacobian
        if self.encoder_hidden is not None and lambda_reg > 0:
            # Compute Jacobian of hidden activations w.r.t. input
            # Sum over all hidden units and batch dimension
            contractive_penalty = 0
            
            for i in range(self.encoder_hidden.size(1)):  # For each hidden unit
                # Compute gradient of i-th hidden unit w.r.t. input
                grad_outputs = torch.zeros_like(self.encoder_hidden)
                grad_outputs[:, i] = 1
                
                gradients = torch.autograd.grad(
                    outputs=self.encoder_hidden,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                # Add squared gradients to penalty
                contractive_penalty += torch.sum(gradients ** 2)
            
            # Normalize by batch size and number of hidden units
            contractive_penalty = contractive_penalty / (x.size(0) * self.encoder_hidden.size(1))
        else:
            contractive_penalty = torch.tensor(0.0, device=x.device)
        
        total_loss = reconstruction_loss + lambda_reg * contractive_penalty
        
        return total_loss, reconstruction_loss, contractive_penalty


class CAEFeatureExtractor:
    """Feature extractor using Contractive Autoencoder."""
    
    def __init__(self, hidden_dim: int = 128, encoding_dim: int = 32, 
                 dropout: float = 0.1, lambda_reg: float = 1e-4):
        """
        Initialize CAE feature extractor.
        
        Args:
            hidden_dim: Hidden layer dimension
            encoding_dim: Encoding dimension (output features)
            dropout: Dropout rate
            lambda_reg: Contractive penalty coefficient
        """
        self.hidden_dim = hidden_dim
        self.encoding_dim = encoding_dim
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def _prepare_data(self, features: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray]:
        """Prepare features for training."""
        # Scale features to [0, 1]
        features_scaled = self.scaler.fit_transform(features.values)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        return features_tensor, features_scaled
    
    def train(self, features: pd.DataFrame, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 1e-3, verbose: bool = True) -> Dict:
        """
        Train the Contractive Autoencoder.
        
        Args:
            features: Input features DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        print(f"Training CAE on {features.shape[0]} samples with {features.shape[1]} features")
        
        # Prepare data
        features_tensor, _ = self._prepare_data(features)
        
        # Initialize model
        input_dim = features.shape[1]
        self.model = ContractiveAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            encoding_dim=self.encoding_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'contractive_penalty': []
        }
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_penalty = 0
            
            for batch_idx, (batch_x,) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                encoded, reconstructed = self.model(batch_x)
                
                # Compute loss
                total_loss, recon_loss, penalty = self.model.contractive_loss(
                    batch_x, reconstructed, self.lambda_reg
                )
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_penalty += penalty.item()
            
            # Average losses
            avg_total_loss = epoch_total_loss / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_penalty = epoch_penalty / len(dataloader)
            
            history['total_loss'].append(avg_total_loss)
            history['reconstruction_loss'].append(avg_recon_loss)
            history['contractive_penalty'].append(avg_penalty)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Total Loss: {avg_total_loss:.6f}, "
                      f"Recon Loss: {avg_recon_loss:.6f}, "
                      f"Penalty: {avg_penalty:.6f}")
        
        print("CAE training completed!")
        return history
    
    def extract_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Extract encoded features using trained CAE.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            Encoded features array
        """
        if self.model is None:
            raise ValueError("Model must be trained before extracting features")
        
        # Scale features using fitted scaler
        features_scaled = self.scaler.transform(features.values)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Extract features
        self.model.eval()
        with torch.no_grad():
            encoded_features, _ = self.model(features_tensor)
            encoded_features = encoded_features.cpu().numpy()
        
        return encoded_features
    
    def save_model(self, filepath: str) -> None:
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'hidden_dim': self.hidden_dim,
            'encoding_dim': self.encoding_dim,
            'dropout': self.dropout,
            'lambda_reg': self.lambda_reg,
            'input_dim': self.model.input_dim
        }
        
        torch.save(save_dict, filepath)
        print(f"CAE model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model and scaler."""
        save_dict = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        self.hidden_dim = save_dict['hidden_dim']
        self.encoding_dim = save_dict['encoding_dim']
        self.dropout = save_dict['dropout']
        self.lambda_reg = save_dict['lambda_reg']
        input_dim = save_dict['input_dim']
        
        # Recreate model
        self.model = ContractiveAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            encoding_dim=self.encoding_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Load state
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.scaler = save_dict['scaler']
        
        print(f"CAE model loaded from {filepath}")


if __name__ == "__main__":
    # Test CAE with synthetic data
    print("Testing Contractive Autoencoder...")
    
    # Create synthetic feature data
    n_samples = 1000
    n_features = 40  # Similar to TA-Lib indicators
    
    np.random.seed(42)
    
    # Generate correlated features
    base_features = np.random.randn(n_samples, 10)
    
    # Create expanded feature set with correlations
    features_data = []
    for i in range(n_features):
        if i < 10:
            features_data.append(base_features[:, i])
        else:
            # Create derived features
            idx1, idx2 = np.random.choice(10, 2, replace=False)
            noise = 0.1 * np.random.randn(n_samples)
            derived = 0.7 * base_features[:, idx1] + 0.3 * base_features[:, idx2] + noise
            features_data.append(derived)
    
    features_array = np.column_stack(features_data)
    features_df = pd.DataFrame(features_array, columns=[f'feature_{i}' for i in range(n_features)])
    
    print(f"Synthetic features shape: {features_df.shape}")
    
    # Initialize and train CAE
    cae = CAEFeatureExtractor(
        hidden_dim=64,
        encoding_dim=16,
        dropout=0.1,
        lambda_reg=1e-4
    )
    
    # Train the model
    history = cae.train(features_df, epochs=50, batch_size=64, verbose=True)
    
    # Extract features
    encoded_features = cae.extract_features(features_df)
    print(f"Encoded features shape: {encoded_features.shape}")
    
    # Show compression ratio
    compression_ratio = features_df.shape[1] / encoded_features.shape[1]
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    print("CAE testing completed successfully!")