# PLSTM-TAL Stock Market Prediction

Implementation of "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL" from scratch.

## Overview

This repository implements the complete pipeline described in the research paper, including:

- **Data Processing**: S&P 500 data collection with fallback to realistic synthetic data
- **Feature Engineering**: 35+ technical indicators (simplified TA-Lib implementation)
- **EEMD Denoising**: Ensemble Empirical Mode Decomposition for noise removal
- **CAE Feature Extraction**: Contractive Autoencoder with penalty term
- **PLSTM-TAL Model**: Peephole LSTM with Temporal Attention Layer
- **Baseline Models**: CNN, LSTM, SVM, Random Forest for comparison
- **Comprehensive Evaluation**: All metrics from the paper (Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC, MCC)

## Project Structure

```
xdj/
├── main.py                    # Main execution script
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── instructions.txt          # Original paper instructions
├── src/                      # Source code modules
│   ├── io.py                 # Data loading and caching
│   ├── indicators.py         # Technical indicators computation
│   ├── eemd.py              # EEMD decomposition implementation
│   ├── cae.py               # Contractive Autoencoder
│   ├── model_plstm_tal.py   # PLSTM-TAL model implementation
│   ├── baselines.py         # Baseline model implementations
│   ├── train.py             # Training infrastructure
│   └── eval.py              # Evaluation and metrics
├── data/                     # Data directory
│   └── sp500_raw.csv        # Cached S&P 500 data
└── results/                  # Output directory
    ├── metrics.json         # Model performance metrics
    ├── model_comparison.csv # Comparison table
    ├── *.png               # Visualization plots
    └── *.pth               # Saved model weights
```

## Key Features Implemented

### 1. Data Pipeline
- **Real Data**: S&P 500 data collection via yfinance (2005-2022)
- **Synthetic Fallback**: Realistic synthetic data generation when real data unavailable
- **Technical Indicators**: 35+ indicators including SMA, EMA, RSI, MACD, Bollinger Bands
- **Data Preprocessing**: Min-max scaling, sequence generation, train/val/test splits

### 2. EEMD Denoising
- **Custom EEMD Implementation**: Ensemble Empirical Mode Decomposition
- **Sample Entropy Calculation**: IMF complexity measurement
- **Noise Removal**: Subtract highest Sample Entropy IMF as per paper methodology

### 3. Contractive Autoencoder (CAE)
- **Contractive Loss**: L_CAE = MSE + λ||J_h(X)||_F^2
- **Jacobian Penalty**: Frobenius norm of encoder Jacobian for regularization
- **Feature Compression**: 35 indicators → 16 latent features
- **Robust Features**: Noise-invariant representations

### 4. PLSTM-TAL Architecture
- **Peephole LSTM**: Custom implementation with cell-to-gate connections
- **Temporal Attention**: Weighted aggregation of hidden states over time
- **Binary Classification**: Stock direction prediction (up/down)
- **Paper Defaults**: 64 units, tanh activation, Adamax optimizer, 0.1 dropout

### 5. Baseline Models
- **CNN**: Simple convolutional network with multiple filter sizes
- **LSTM**: Standard LSTM without peepholes or attention
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: Ensemble of decision trees

### 6. Evaluation Framework
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC, MCC
- **Visualizations**: Confusion matrices, ROC curves, PR curves
- **Model Comparison**: Side-by-side performance analysis
- **Results Export**: JSON metrics, CSV comparisons, saved models

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ajaygm18/xdj.git
cd xdj
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
Run the complete pipeline with default settings:
```bash
python main.py
```

### Custom Configuration
Modify `config.json` or provide your own:
```bash
python main.py --config my_config.json --output my_results/
```

### Configuration Options

```json
{
  "cae": {
    "hidden_dim": 64,
    "encoding_dim": 16,
    "dropout": 0.1,
    "lambda_reg": 1e-4,
    "epochs": 50
  },
  "plstm_tal": {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.1
  },
  "training": {
    "window_length": 20,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "patience": 10
  }
}
```

## Results

The implementation achieves the following performance on synthetic S&P 500 data:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | PR-AUC | MCC |
|-------|----------|-----------|--------|----------|---------|--------|-----|
| **PLSTM-TAL** | 0.471 | 0.467 | 0.401 | 0.432 | 0.458 | 0.472 | -0.059 |
| CNN | 0.471 | 0.470 | 0.444 | 0.457 | 0.443 | 0.463 | -0.058 |
| LSTM | 0.447 | 0.453 | 0.505 | 0.478 | 0.446 | 0.467 | -0.108 |
| SVM | 0.445 | 0.442 | 0.409 | 0.425 | **0.583** | **0.560** | -0.110 |
| Random Forest | 0.433 | 0.428 | 0.391 | 0.408 | 0.401 | 0.440 | -0.135 |

### Key Observations
- **PLSTM-TAL** achieves competitive performance with the baseline models
- **SVM** shows strongest AUC performance, indicating good probability calibration
- All models show similar accuracy levels (~44-47%), suggesting the prediction task is challenging
- Results include comprehensive visualizations (confusion matrices, ROC/PR curves)

## Technical Implementation Details

### Model Architecture
The PLSTM-TAL model implements:

1. **Peephole LSTM Cells**: 
   ```
   f_t = σ(W_f [x_t, h_{t-1}] + w_cf * c_{t-1} + b_f)
   i_t = σ(W_i [x_t, h_{t-1}] + w_ci * c_{t-1} + b_i)  
   o_t = σ(W_o [x_t, h_{t-1}] + w_co * c_t + b_o)
   ```

2. **Temporal Attention**:
   ```
   α_t = softmax(w_att^T * tanh(W_att * h_t))
   context = Σ(α_t * h_t)
   ```

3. **Contractive Loss**:
   ```
   L_CAE = MSE(x, x_hat) + λ * ||∇_x h(x)||_F^2
   ```

### Data Processing Pipeline
1. **Feature Generation**: Technical indicators + LOG_RETURN
2. **EEMD Filtering**: Remove highest Sample Entropy IMF
3. **CAE Encoding**: Compress features 35→16 dimensions
4. **Sequence Creation**: Sliding windows of length 20
5. **Label Generation**: Binary direction based on returns

## Dependencies

- **Python**: 3.10+
- **PyTorch**: 2.0+ (deep learning framework)
- **scikit-learn**: 1.3+ (traditional ML models)
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **yfinance**: Stock data collection

## Limitations and Future Work

### Current Limitations
1. **TA-Lib Dependency**: Uses simplified indicators instead of full TA-Lib
2. **Signal-EMD**: Custom EEMD implementation instead of signal-emd library
3. **Synthetic Data**: Fallback due to network restrictions in environment
4. **Single Market**: Currently S&P 500 only (can be extended)

### Future Enhancements
1. **Real Data Integration**: Direct connection to financial data providers
2. **Multi-Market Support**: FTSE, SSE, Nifty 50 as in original paper
3. **Bayesian Optimization**: Hyperparameter tuning implementation
4. **Real-time Prediction**: Live trading signal generation
5. **Advanced Indicators**: Full TA-Lib integration

## Paper Reference

Original paper: "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL in urbanized smart cities"

This implementation follows the paper's methodology with practical adaptations for reproducibility and ease of use.

## License

This project is for educational and research purposes. Please cite the original paper when using this implementation.

---

**Note**: This implementation is designed to be self-contained and reproducible. All major components are implemented from scratch to ensure transparency and educational value.