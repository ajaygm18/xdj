# PLSTM-TAL Stock Market Prediction

Implementation of "Enhanced prediction of stock markets using a novel deep learning model PLSTM-TAL" with complete technical indicators suite and comprehensive evaluation framework.

## Overview

This repository implements the complete PLSTM-TAL (Peephole LSTM with Temporal Attention Layer) pipeline for stock market trend prediction, as described in the research paper. The implementation includes all 40 technical indicators, EEMD denoising, contractive autoencoder, and comprehensive baseline comparisons.

### Key Features Implemented

#### 1. Complete Technical Indicators Suite
- **All 40 Indicators**: Bollinger Bands, WMA, EMA, DEMA, KAMA, MAMA, MIDPRICE, SAR, SMA, T3, TEMA, TRIMA, AD, ADOSC, OBV, MEDPRICE, TYPPRICE, WCLPRICE, ADX, ADXR, APO, AROON, AROONOSC, BOP, CCI, CMO, DX, MACD, MFI, MINUS_DI, MOM, PLUS_DI, LOG_RETURN, PPO, ROC, RSI, STOCH, STOCHRSI, ULTOSC, WILLR
- **Paper-Compliant Implementation**: All indicators computed using TA-Lib compatible methods
- **Robust Feature Engineering**: Proper handling of NaN values and normalization

#### 2. Data Pipeline
- **Real Market Data**: S&P 500 data from Yahoo Finance (2015-2025)
- **Data Validation**: Outlier detection and data quality checks
- **Synthetic Fallback**: Realistic synthetic data generation when needed

#### 3. EEMD Denoising
- **Ensemble Empirical Mode Decomposition**: Custom implementation with configurable parameters
- **Sample Entropy Calculation**: IMF complexity measurement for noise identification
- **Adaptive Noise Removal**: Removes highest entropy IMF for signal enhancement

#### 4. Contractive Autoencoder (CAE)
- **Feature Compression**: 46 indicators → 32 latent features
- **Contractive Loss**: L_CAE = MSE + λ||J_h(X)||_F^2
- **Jacobian Penalty**: Regularization for noise-invariant representations

#### 5. PLSTM-TAL Architecture
- **Peephole LSTM**: Enhanced LSTM with peephole connections
- **Temporal Attention Layer**: Attention mechanism for sequence modeling
- **Paper-Compliant Hyperparameters**: Units=64, Activation=tanh, Optimizer=Adamax

#### 6. Comprehensive Baseline Models
- **CNN**: Convolutional Neural Network with optimized architecture
- **LSTM**: Standard LSTM baseline
- **SVM**: Support Vector Machine with RBF kernel
- **Random Forest**: Ensemble tree-based method

#### 7. Evaluation Framework
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, PR-AUC, MCC
- **Visualization**: ROC curves, PR curves, confusion matrices
- **Model Comparison**: Comprehensive performance analysis

## Project Structure

```
xdj/
├── main.py                    # Main execution script
├── config.json               # Full configuration
├── config_fast.json          # Fast testing configuration
├── requirements.txt          # Python dependencies
├── instructions.txt          # Original paper instructions
├── src/                      # Source code modules
│   ├── io.py                 # Data loading and caching
│   ├── indicators.py         # 40+ technical indicators implementation
│   ├── eemd.py              # EEMD decomposition implementation
│   ├── cae.py               # Contractive Autoencoder
│   ├── model_plstm_tal.py   # PLSTM-TAL model implementation
│   ├── baselines.py         # Baseline model implementations
│   ├── train.py             # Training infrastructure
│   ├── eval.py              # Evaluation and metrics
│   └── data_loader.py       # Data preprocessing utilities
├── data/                     # Data directory
│   └── sp500_raw.csv        # Cached S&P 500 data
├── results/                  # Full pipeline results
│   ├── metrics.json         # Model performance metrics
│   ├── model_comparison.csv # Comparison table
│   ├── *.png               # Visualization plots
│   └── *.pth               # Saved model weights
└── results_fast/            # Fast configuration results
    ├── metrics.json         # Model performance metrics
    ├── model_comparison.csv # Comparison table
    ├── *.png               # Visualization plots
    └── *.pth               # Saved model weights
```

## Installation and Setup

### Prerequisites
- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ajaygm18/xdj.git
cd xdj

# Install Python dependencies
pip install -r requirements.txt
```

**Note**: TA-Lib requires the C library to be installed first:
- **macOS**: `brew install ta-lib`
- **Ubuntu/Debian**: `sudo apt-get install libta-lib-dev`
- **Windows**: Download from [TA-Lib website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

## Usage

### Quick Start (Fast Configuration)

For quick testing and validation:

```bash
python main.py --config config_fast.json
```

This runs with reduced epochs and ensembles for faster execution (~15 minutes).

### Full Pipeline (Production Configuration)

For best results matching the paper:

```bash
python main.py --config config.json
```

This runs the full pipeline with optimized parameters (~2-3 hours).

### Custom Configuration

You can modify `config.json` or create your own configuration file:

```json
{
  "data_dir": "data",
  "results_dir": "results",
  "symbol": "^GSPC",
  "data_years": 10,
  "cae": {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-3
  },
  "training": {
    "epochs": 300,
    "batch_size": 32,
    "patience": 50
  },
  "eemd": {
    "n_ensembles": 100,
    "noise_scale": 0.2
  }
}
```

## Results

### Model Performance (Fast Configuration)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | PR-AUC | MCC |
|-------|----------|-----------|---------|----------|---------|--------|-----|
| **LSTM** | **72.19%** | **71.93%** | **68.72%** | **70.29%** | **80.75%** | **82.32%** | **44.22%** |
| **SVM** | **72.19%** | 74.51% | 63.69% | 68.67% | 77.46% | 78.40% | 44.39% |
| **Random Forest** | 71.12% | 73.20% | 62.57% | 67.47% | 75.16% | 73.43% | 42.21% |
| CNN | 51.07% | 45.83% | 12.29% | 19.38% | 45.34% | 46.08% | -1.56% |
| PLSTM-TAL | 49.47% | 43.24% | 17.88% | 25.30% | 45.36% | 46.14% | -4.59% |

### Key Findings

1. **LSTM Baseline Performance**: Achieved the highest accuracy (72.19%) with excellent AUC-ROC (80.75%)
2. **SVM Strong Performance**: Competitive results with highest precision (74.51%)
3. **PLSTM-TAL**: Underperformed in this configuration, likely due to:
   - Reduced training epochs (100 vs 300)
   - Smaller ensemble size (20 vs 100)
   - Need for hyperparameter tuning

### Visualizations

![Model Comparison](results_fast/model_comparison.png)

#### ROC Curves
- **LSTM**: [ROC Curve](results_fast/lstm_roc_curve.png) - AUC: 80.75%
- **SVM**: [ROC Curve](results_fast/svm_roc_curve.png) - AUC: 77.46%
- **Random Forest**: [ROC Curve](results_fast/random_forest_roc_curve.png) - AUC: 75.16%

#### Precision-Recall Curves
- **LSTM**: [PR Curve](results_fast/lstm_pr_curve.png) - PR-AUC: 82.32%
- **SVM**: [PR Curve](results_fast/svm_pr_curve.png) - PR-AUC: 78.40%
- **Random Forest**: [PR Curve](results_fast/random_forest_pr_curve.png) - PR-AUC: 73.43%

#### Confusion Matrices
- **LSTM**: [Confusion Matrix](results_fast/lstm_confusion_matrix.png)
- **SVM**: [Confusion Matrix](results_fast/svm_confusion_matrix.png)
- **Random Forest**: [Confusion Matrix](results_fast/random_forest_confusion_matrix.png)

## Technical Implementation Details

### Technical Indicators

All 40 indicators from the paper are implemented:

**Overlap Studies:**
- Bollinger Bands (BBANDS) - upper, middle, lower bands
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA, T3, TRIMA
- Adaptive Averages: KAMA, MAMA
- Price Indicators: MIDPRICE, MEDPRICE, TYPPRICE, WCLPRICE
- Parabolic SAR (SAR)

**Volume Indicators:**
- Accumulation/Distribution Line (AD)
- Chaikin A/D Oscillator (ADOSC)
- On Balance Volume (OBV)

**Momentum Indicators:**
- RSI, MACD (with signal and histogram)
- Stochastic (STOCH) - %K and %D
- Stochastic RSI (STOCHRSI)
- Williams' %R (WILLR)
- Ultimate Oscillator (ULTOSC)
- ROC, MOM, PPO, APO

**Trend Indicators:**
- ADX, ADXR, DX
- Directional Movement: PLUS_DI, MINUS_DI
- Aroon (up/down) and Aroon Oscillator

**Custom Indicators:**
- LOG_RETURN: log(P_t / P_{t-1})
- CCI, CMO, BOP, MFI

### Data Processing Pipeline

1. **Data Collection**: Yahoo Finance API for real market data
2. **Feature Generation**: 46 technical indicators (40 from paper + variations)
3. **EEMD Filtering**: Remove highest Sample Entropy IMF
4. **CAE Encoding**: Compress features 46→32 dimensions
5. **Sequence Creation**: Sliding windows of length 20
6. **Label Generation**: Binary direction based on returns

### Model Architecture

**PLSTM-TAL Components:**
- Peephole LSTM with forget, input, and output gates
- Temporal attention mechanism
- Dense output layer with tanh activation
- Binary classification with sigmoid output

**Training Configuration:**
- Optimizer: Adamax (learning_rate=1e-3)
- Loss: Binary crossentropy
- Regularization: Dropout (0.1)
- Early stopping with patience

## Configuration Options

### EEMD Parameters
- `n_ensembles`: Number of ensemble members (20-100)
- `noise_scale`: Noise amplitude (0.1-0.3)
- `max_imfs`: Maximum IMFs to extract

### CAE Parameters
- `encoding_dim`: Latent space dimension (16-64)
- `lambda_reg`: Contractive penalty weight (1e-5 to 1e-3)
- `epochs`: Training epochs (50-200)

### Training Parameters
- `window_length`: Sequence length (10-30)
- `batch_size`: Mini-batch size (16-64)
- `patience`: Early stopping patience (20-50)

## Troubleshooting

### Common Issues

1. **TA-Lib Installation Error**
   - Install C library first: `brew install ta-lib` (macOS)
   - Windows: Use pre-compiled wheels

2. **Memory Issues with EEMD**
   - Reduce `n_ensembles` in config
   - Use smaller data windows

3. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Use CPU: Add `device='cpu'` to config

4. **Poor Model Performance**
   - Increase training epochs
   - Tune learning rate
   - Check data quality and feature scaling

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{plstm_tal_2024,
  title={PLSTM-TAL: Peephole LSTM with Temporal Attention Layer for Stock Market Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## Acknowledgments

- TA-Lib library for technical indicators
- Yahoo Finance for market data
- PyTorch team for deep learning framework
- Research paper authors for methodology