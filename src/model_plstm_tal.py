"""
PLSTM-TAL model implementation: Peephole LSTM with Temporal Attention Layer.
Implements the core deep learning model from the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PeepholeLSTMCell(nn.Module):
    """
    LSTM Cell with peephole connections.
    
    Peephole connections allow gates to access the cell state:
    - Forget gate: f_t = σ(W_f [x_t, h_{t-1}] + w_cf * c_{t-1} + b_f)
    - Input gate: i_t = σ(W_i [x_t, h_{t-1}] + w_ci * c_{t-1} + b_i)  
    - Output gate: o_t = σ(W_o [x_t, h_{t-1}] + w_co * c_t + b_o)
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize Peephole LSTM Cell.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            bias: Whether to use bias terms
        """
        super(PeepholeLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden and hidden-to-hidden weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
        # Peephole connections (cell-to-gate weights)
        self.weight_cf = nn.Parameter(torch.randn(hidden_size))  # Cell to forget gate
        self.weight_ci = nn.Parameter(torch.randn(hidden_size))  # Cell to input gate  
        self.weight_co = nn.Parameter(torch.randn(hidden_size))  # Cell to output gate
        
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with improved LSTM initialization."""
        std = 1.0 / np.sqrt(self.hidden_size)
        
        # Initialize weight matrices with Xavier uniform
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        # Initialize peephole connections with smaller values
        nn.init.uniform_(self.weight_cf, -0.1, 0.1)
        nn.init.uniform_(self.weight_ci, -0.1, 0.1) 
        nn.init.uniform_(self.weight_co, -0.1, 0.1)
        
        # Initialize biases properly for LSTM
        if self.bias:
            # Initialize all biases to zero except forget gate bias
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            
            # Set forget gate bias to 1 for better gradient flow
            # Forget gate is the second quarter of the bias vector
            forget_bias_start = self.hidden_size
            forget_bias_end = 2 * self.hidden_size
            nn.init.ones_(self.bias_ih[forget_bias_start:forget_bias_end])
            nn.init.ones_(self.bias_hh[forget_bias_start:forget_bias_end])
    
    def forward(self, input_tensor: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through peephole LSTM cell.
        
        Args:
            input_tensor: Input tensor (batch_size, input_size)
            hidden: Tuple of (h_{t-1}, c_{t-1})
            
        Returns:
            Tuple of (h_t, c_t)
        """
        h_prev, c_prev = hidden
        
        # Linear transformations
        gi = F.linear(input_tensor, self.weight_ih, self.bias_ih)
        gh = F.linear(h_prev, self.weight_hh, self.bias_hh)
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)
        
        # Gates with peephole connections
        forget_gate = torch.sigmoid(i_f + h_f + self.weight_cf * c_prev)
        input_gate = torch.sigmoid(i_i + h_i + self.weight_ci * c_prev)
        candidate = torch.tanh(i_g + h_g)
        
        # Update cell state
        c_new = forget_gate * c_prev + input_gate * candidate
        
        # Output gate with peephole to new cell state
        output_gate = torch.sigmoid(i_o + h_o + self.weight_co * c_new)
        
        # Update hidden state
        h_new = output_gate * torch.tanh(c_new)
        
        return h_new, c_new


class PeepholeLSTM(nn.Module):
    """Multi-layer Peephole LSTM."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 dropout: float = 0.0, bidirectional: bool = False):
        """
        Initialize multi-layer Peephole LSTM.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(PeepholeLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                cell_input_size = input_size
            else:
                cell_input_size = hidden_size * (2 if bidirectional else 1)
            
            if bidirectional:
                self.cells.append(nn.ModuleDict({
                    'forward': PeepholeLSTMCell(cell_input_size, hidden_size),
                    'backward': PeepholeLSTMCell(cell_input_size, hidden_size)
                }))
            else:
                self.cells.append(PeepholeLSTMCell(cell_input_size, hidden_size))
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
    
    def forward(self, input_seq: torch.Tensor, initial_hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass through multi-layer Peephole LSTM.
        
        Args:
            input_seq: Input sequence (seq_len, batch_size, input_size)
            initial_hidden: Initial hidden states
            
        Returns:
            Tuple of (output_seq, final_hidden)
        """
        seq_len, batch_size, _ = input_seq.size()
        
        if initial_hidden is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size,
                             dtype=input_seq.dtype, device=input_seq.device)
            c_0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size,
                             dtype=input_seq.dtype, device=input_seq.device)
            initial_hidden = (h_0, c_0)
        
        outputs = []
        current_input = input_seq
        
        for layer in range(self.num_layers):
            layer_outputs = []
            
            if self.bidirectional:
                # Forward direction
                h_f = initial_hidden[0][layer * 2]
                c_f = initial_hidden[1][layer * 2]
                
                # Backward direction  
                h_b = initial_hidden[0][layer * 2 + 1]
                c_b = initial_hidden[1][layer * 2 + 1]
                
                forward_outputs = []
                backward_outputs = []
                
                # Forward pass
                for t in range(seq_len):
                    h_f, c_f = self.cells[layer]['forward'](current_input[t], (h_f, c_f))
                    forward_outputs.append(h_f)
                
                # Backward pass
                for t in reversed(range(seq_len)):
                    h_b, c_b = self.cells[layer]['backward'](current_input[t], (h_b, c_b))
                    backward_outputs.insert(0, h_b)
                
                # Concatenate forward and backward outputs
                for t in range(seq_len):
                    layer_outputs.append(torch.cat([forward_outputs[t], backward_outputs[t]], dim=1))
                
            else:
                # Unidirectional
                h = initial_hidden[0][layer]
                c = initial_hidden[1][layer]
                
                for t in range(seq_len):
                    h, c = self.cells[layer](current_input[t], (h, c))
                    layer_outputs.append(h)
            
            # Stack outputs for this layer
            layer_output = torch.stack(layer_outputs, dim=0)
            
            # Apply dropout between layers
            if self.dropout_layer is not None and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            
            current_input = layer_output
            outputs.append(layer_output)
        
        # Return final layer output and final hidden states
        final_hidden = (h, c) if not self.bidirectional else ((h_f, h_b), (c_f, c_b))
        
        return outputs[-1], final_hidden


class TemporalAttentionLayer(nn.Module):
    """
    Temporal Attention Layer for weighting time steps.
    
    Computes attention weights over the temporal dimension and produces
    a context vector as weighted sum of hidden states.
    """
    
    def __init__(self, hidden_size: int, attention_dim: int = None):
        """
        Initialize Temporal Attention Layer.
        
        Args:
            hidden_size: Hidden state dimension
            attention_dim: Attention computation dimension
        """
        super(TemporalAttentionLayer, self).__init__()
        
        if attention_dim is None:
            attention_dim = hidden_size
        
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        
        # Attention mechanism components
        self.attention_linear = nn.Linear(hidden_size, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal attention.
        
        Args:
            hidden_states: Hidden states (seq_len, batch_size, hidden_size)
            
        Returns:
            Tuple of (context_vector, attention_weights)
        """
        seq_len, batch_size, hidden_size = hidden_states.size()
        
        # Compute attention scores
        # h_att = tanh(W_att * h_t + b_att)
        attention_hidden = torch.tanh(self.attention_linear(hidden_states))  # (seq_len, batch_size, attention_dim)
        
        # e_t = w_att^T * h_att_t
        attention_scores = self.context_vector(attention_hidden).squeeze(-1)  # (seq_len, batch_size)
        
        # Apply softmax to get attention weights
        # α_t = softmax(e_t)
        attention_weights = F.softmax(attention_scores, dim=0)  # (seq_len, batch_size)
        
        # Compute context vector as weighted sum
        # c = Σ(α_t * h_t)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=0)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights


class PLSTM_TAL(nn.Module):
    """
    PLSTM-TAL: Peephole LSTM with Temporal Attention Layer.
    
    Implements the exact architecture from the paper diagram:
    1. Peephole LSTM for sequence modeling
    2. Temporal Attention Layer for weighted feature aggregation
    3. Batch Normalization for stable training
    4. Flatten layer for dimension consistency
    5. Multiple Dense layers with dropout for classification
    
    Architecture Flow (following diagram):
    Input -> Peephole LSTM -> Temporal Attention -> Batch Norm -> 
    Flatten -> Dense -> Dropout -> Dense -> Dropout -> Output
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.1, bidirectional: bool = False, 
                 attention_dim: int = None, activation: str = 'tanh'):
        """
        Initialize PLSTM-TAL model.
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden state dimension (default: 64 from paper)
            num_layers: Number of LSTM layers
            dropout: Dropout rate (default: 0.1 from paper)
            bidirectional: Whether to use bidirectional LSTM
            attention_dim: Attention mechanism dimension
            activation: Activation function for LSTM ('tanh' from paper)
        """
        super(PLSTM_TAL, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Ensure tanh activation as specified in paper
        if activation != 'tanh':
            print(f"Warning: Paper specifies tanh activation, but {activation} was requested")
        
        # Peephole LSTM layers
        self.plstm = PeepholeLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Temporal Attention Layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.temporal_attention = TemporalAttentionLayer(
            hidden_size=lstm_output_size,
            attention_dim=attention_dim
        )
        
        # Classification head - Following exact architecture from diagram
        # The diagram shows: Batch Norm -> Flatten -> Dense -> Dropout -> Dense -> Dropout -> Output
        self.batch_norm = nn.BatchNorm1d(lstm_output_size)
        
        # Flatten layer (context vector is already flattened, but keeping for architecture compliance)
        self.flatten = nn.Flatten()
        
        # Multiple dense layers as shown in diagram
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),  # First dense layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),  # Second dense layer  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1),  # Output layer for binary classification
        )
        
        # Initialize classifier layers properly
        self._init_classifier_weights()
        
    def _init_classifier_weights(self):
        """Initialize classifier weights properly."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Initialize attention layer weights
        nn.init.xavier_uniform_(self.temporal_attention.attention_linear.weight)
        if self.temporal_attention.attention_linear.bias is not None:
            nn.init.zeros_(self.temporal_attention.attention_linear.bias)
        nn.init.xavier_uniform_(self.temporal_attention.context_vector.weight)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through PLSTM-TAL following exact architecture from diagram.
        
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # Reshape input: (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
        x = x.transpose(0, 1)
        
        # Pass through Peephole LSTM
        lstm_output, _ = self.plstm(x)  # (seq_len, batch_size, hidden_size)
        
        # Apply Temporal Attention
        context_vector, attention_weights = self.temporal_attention(lstm_output)
        
        # Apply Batch Normalization as shown in diagram
        context_vector = self.batch_norm(context_vector)
        
        # Flatten (already flat but following diagram)
        context_vector = self.flatten(context_vector)
        
        # Classification through multiple dense layers
        logits = self.classifier(context_vector).squeeze(-1)  # (batch_size,)
        
        return logits, attention_weights


if __name__ == "__main__":
    # Test PLSTM-TAL model
    print("Testing PLSTM-TAL model...")
    
    # Model parameters (from paper defaults)
    batch_size = 32
    seq_len = 20  # Window length
    input_size = 48  # CAE features (16) + filtered price (1) + other features
    hidden_size = 64  # From paper
    dropout = 0.1  # From paper
    
    # Create synthetic input data
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, input_size)
    
    print(f"Input shape: {x.shape}")
    
    # Initialize model
    model = PLSTM_TAL(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=dropout,
        bidirectional=False,
        activation='tanh'
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(x)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)
    print(f"Prediction probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    # Test with different sequence lengths
    for test_seq_len in [10, 15, 25]:
        x_test = torch.randn(batch_size, test_seq_len, input_size)
        with torch.no_grad():
            logits_test, attention_test = model(x_test)
        print(f"Seq len {test_seq_len}: Output shape {logits_test.shape}, Attention shape {attention_test.shape}")
    
    print("PLSTM-TAL model testing completed successfully!")