# PLSTM-TAL Stock Market Prediction - Streamlit Web UI

## Overview

This repository now includes a **Streamlit web interface** that allows users to predict stock market trends for **any custom stock symbol** using the PLSTM-TAL (Peephole LSTM with Temporal Attention Layer) model. The web interface provides an intuitive way to:

- Select any stock symbol from Yahoo Finance
- Configure model parameters (with paper-compliant defaults)
- Train the PLSTM-TAL model with real-time progress tracking
- View accuracy metrics and predictions
- Compare results with baseline models

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Web Interface Features

### Stock Selection
- **Popular Symbols**: Quick selection from curated lists (US Large Cap, Indices, Indian Stocks, etc.)
- **Custom Symbol**: Enter any valid Yahoo Finance symbol (e.g., AAPL, TSLA, GOOGL, RELIANCE.NS, BTC-USD)

### Date Range Options
- **Paper Default**: 2005-01-01 to 2022-03-31 (original research timeframe)
- **Preset Ranges**: Last 1, 3, or 5 years
- **Custom Range**: Select any date range

### Model Configuration
- **Paper-Compliant Parameters**: Use exact parameters from the research paper
- **Custom Parameters**: Adjust hidden size, dropout, learning rate, etc.
- **Advanced Options**: Enable Bayesian optimization, quick mode, model saving

### Real-time Training
- Progress tracking with 6-step pipeline
- Live status updates during training
- Early stopping and validation monitoring

### Results Dashboard
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Model Comparison**: PLSTM-TAL vs baseline models (LSTM, SVM)
- **Visualizations**: Performance charts and comparisons
- **Export Results**: Download metrics as JSON

## 🔧 Technical Implementation

### New Components

1. **`CustomStockDataLoader`** (`src/custom_stock_loader.py`)
   - Supports any Yahoo Finance stock symbol
   - Automatic currency detection
   - Data validation and quality checks
   - Popular symbols categorization

2. **Streamlit UI** (`streamlit_app.py`)
   - Modern, responsive web interface
   - Real-time progress tracking
   - Interactive parameter configuration
   - Results visualization and export

3. **Integration Tests** (`test_integration.py`, `quick_test.py`)
   - End-to-end pipeline validation
   - Performance testing with different stocks

### Architecture

```
Streamlit UI
    ↓
CustomStockDataLoader → TechnicalIndicators → EEMD → CAE → PLSTM-TAL
    ↓                                                        ↓
YFinance API                                          Model Training & Evaluation
```

## 📊 Supported Stock Symbols

The system supports any valid Yahoo Finance symbol, including:

- **US Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, etc.
- **Indices**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ)
- **International**: RELIANCE.NS (India), ASML.AS (Netherlands), SAP.DE (Germany)
- **Crypto**: BTC-USD, ETH-USD, ADA-USD
- **Commodities**: GC=F (Gold), CL=F (Oil), SI=F (Silver)

## 🎯 Model Parameters (Paper-Compliant)

When "Use Paper-Compliant Parameters" is enabled:

- **Hidden Size**: 64
- **Dropout**: 0.1
- **Activation**: tanh
- **Optimizer**: Adamax
- **Window Length**: 20
- **Batch Size**: 32
- **Learning Rate**: 1e-3

## 📈 Example Usage

1. **Open the app**: `streamlit run streamlit_app.py`
2. **Select a stock**: Choose "TSLA" from US Large Cap or enter a custom symbol
3. **Configure dates**: Use "Last 3 Years" for recent data
4. **Load data**: Click "🔄 Load Stock Data" to fetch and analyze
5. **Train model**: Go to "Model Training" tab and click "🚀 Train PLSTM-TAL Model"
6. **View results**: Check "Predictions" and "Results" tabs for metrics and comparisons

## 🔍 Testing

Run the integration tests to verify everything works:

```bash
# Test core components
python test_integration.py

# Test quick training pipeline
python quick_test.py
```

## 🆚 Original vs New Implementation

| Feature | Original | New Streamlit UI |
|---------|----------|------------------|
| Stock Support | USA (S&P 500) + Indian (RELIANCE, IRFC) | Any Yahoo Finance symbol |
| Interface | Command-line | Web-based GUI |
| User Experience | Technical users | General users |
| Parameter Config | Code editing | Interactive UI |
| Progress Tracking | Console logs | Real-time progress bars |
| Results View | Text output | Interactive charts |
| Stock Selection | Hardcoded | Dynamic with validation |

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- yfinance 0.2+
- All other dependencies in `requirements.txt`

## 🎉 Key Achievements

✅ **Universal Stock Support**: Can now predict any stock symbol instead of just USA stocks
✅ **User-Friendly Interface**: Streamlit web UI replaces command-line interface  
✅ **Real-time Training**: Live progress tracking and status updates
✅ **Model Comparison**: Side-by-side comparison with baseline models
✅ **Paper Compliance**: Maintains original research parameters as defaults
✅ **Data Validation**: Automatic symbol validation and data quality checks
✅ **Export Functionality**: Download results for further analysis

## 🚀 Future Enhancements

Potential improvements for the web interface:
- Real-time streaming predictions
- Portfolio analysis with multiple stocks
- Advanced visualization with technical indicators
- Model performance monitoring over time
- Integration with trading APIs
- Mobile-responsive design improvements

---

The Streamlit web interface makes the PLSTM-TAL stock prediction model accessible to a broader audience while maintaining the technical rigor of the original research implementation.