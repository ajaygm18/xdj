"""
Enhanced PLSTM-TAL model implementation with advanced architecture for high accuracy.
Includes multi-head attention, residual connections, and sophisticated feature processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from model_plstm_tal import PeepholeLSTMCell, PeepholeLSTM


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-Head Temporal Attention Layer for enhanced sequence modeling.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize Multi-Head Temporal Attention.
        
        Args:
            hidden_size: Hidden state dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadTemporalAttention, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-head temporal attention.
        
        Args:
            hidden_states: Hidden states (seq_len, batch_size, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        seq_len, batch_size, hidden_size = hidden_states.size()
        
        # Transpose for attention computation: (batch_size, seq_len, hidden_size)
        hidden_states = hidden_states.transpose(0, 1)
        
        # Compute Q, K, V
        Q = self.query(hidden_states)  # (batch_size, seq_len, hidden_size)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection
        context = self.output_projection(context)
        
        # Residual connection and layer norm
        context = self.layer_norm(context + hidden_states)
        
        # Global attention pooling - weighted average over time steps
        temporal_weights = F.softmax(torch.mean(attention_weights, dim=1), dim=-1)  # Average over heads
        temporal_weights = torch.mean(temporal_weights, dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        
        # Weighted sum for final context vector
        context_vector = torch.sum(temporal_weights * context, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, temporal_weights.squeeze(-1)


class EnhancedFeatureProcessor(nn.Module):
    """
    Enhanced feature processor with multiple pathways for different feature types.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.2):
        """
        Initialize Enhanced Feature Processor.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden processing dimension
            dropout: Dropout rate
        """
        super(EnhancedFeatureProcessor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Feature pathway networks
        self.technical_pathway = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.price_pathway = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through multiple pathways.
        
        Args:
            x: Input features (batch_size, seq_len, input_size)
            
        Returns:
            Enhanced features (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, input_size = x.size()
        
        # Reshape for pathway processing
        x_flat = x.view(-1, input_size)
        
        # Process through pathways
        technical_features = self.technical_pathway(x_flat)
        price_features = self.price_pathway(x_flat)
        
        # Concatenate pathway outputs
        combined_features = torch.cat([technical_features, price_features], dim=-1)
        
        # Fusion
        enhanced_features = self.fusion(combined_features)
        
        # Reshape back to sequence format
        enhanced_features = enhanced_features.view(batch_size, seq_len, self.hidden_size)
        
        return enhanced_features


class EnhancedPLSTM_TAL(nn.Module):
    """
    Enhanced PLSTM-TAL: Advanced Peephole LSTM with Multi-Head Temporal Attention.
    
    Key enhancements:
    1. Multi-head temporal attention for better sequence modeling
    2. Bidirectional processing for richer representations
    3. Residual connections for better gradient flow
    4. Enhanced feature processing pathways
    5. Layer normalization for training stability
    6. Advanced regularization techniques
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 dropout: float = 0.3, bidirectional: bool = True, 
                 attention_heads: int = 8, attention_dropout: float = 0.2,
                 use_residual: bool = True, use_layer_norm: bool = True,
                 activation: str = 'tanh'):
        """
        Initialize Enhanced PLSTM-TAL model.
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            attention_heads: Number of attention heads
            attention_dropout: Attention dropout rate
            use_residual: Whether to use residual connections
            use_layer_norm: Whether to use layer normalization
            activation: Activation function for LSTM
        """
        super(EnhancedPLSTM_TAL, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Enhanced feature processor
        self.feature_processor = EnhancedFeatureProcessor(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        # Peephole LSTM layers
        self.plstm = PeepholeLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Multi-Head Temporal Attention Layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.multi_head_attention = MultiHeadTemporalAttention(
            hidden_size=lstm_output_size,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(lstm_output_size)
            self.layer_norm2 = nn.LayerNorm(lstm_output_size)
        
        # Enhanced classification head with multiple stages
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.BatchNorm1d(lstm_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.BatchNorm1d(lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.BatchNorm1d(lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(lstm_output_size // 4, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Enhanced PLSTM-TAL.
        
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Enhanced feature processing
        enhanced_features = self.feature_processor(x)
        
        # Transpose for LSTM: (seq_len, batch_size, hidden_size)
        enhanced_features = enhanced_features.transpose(0, 1)
        
        # Pass through Peephole LSTM
        lstm_output, _ = self.plstm(enhanced_features)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            lstm_output = self.layer_norm1(lstm_output.transpose(0, 1)).transpose(0, 1)
        
        # Apply Multi-Head Temporal Attention
        context_vector, attention_weights = self.multi_head_attention(lstm_output)
        
        # Apply layer normalization if enabled
        if self.use_layer_norm:
            context_vector = self.layer_norm2(context_vector)
        
        # Classification
        logits = self.classifier(context_vector).squeeze(-1)
        
        return logits, attention_weights


def create_enhanced_model(config: dict) -> EnhancedPLSTM_TAL:
    """
    Create Enhanced PLSTM-TAL model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured Enhanced PLSTM-TAL model
    """
    plstm_config = config.get('plstm_tal', {})
    
    model = EnhancedPLSTM_TAL(
        input_size=config.get('input_size', 40),
        hidden_size=plstm_config.get('hidden_size', 128),
        num_layers=plstm_config.get('num_layers', 3),
        dropout=plstm_config.get('dropout', 0.3),
        bidirectional=plstm_config.get('bidirectional', True),
        attention_heads=plstm_config.get('attention_heads', 8),
        attention_dropout=plstm_config.get('attention_dropout', 0.2),
        use_residual=plstm_config.get('use_residual', True),
        use_layer_norm=plstm_config.get('use_layer_norm', True),
        activation=plstm_config.get('activation', 'tanh')
    )
    
    return model


if __name__ == "__main__":
    # Test Enhanced PLSTM-TAL model
    print("Testing Enhanced PLSTM-TAL model...")
    
    # Model parameters
    batch_size = 64
    seq_len = 30
    input_size = 40
    hidden_size = 128
    dropout = 0.3
    
    # Create synthetic input data
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"Input shape: {x.shape}")
    
    # Initialize enhanced model
    model = EnhancedPLSTM_TAL(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=3,
        dropout=dropout,
        bidirectional=True,
        attention_heads=8,
        activation='tanh'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(x)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)
    print(f"Prediction probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    print("Enhanced PLSTM-TAL model testing completed successfully!")