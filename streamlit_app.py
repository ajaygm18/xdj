"""
Streamlit UI for PLSTM-TAL Stock Market Prediction
Allows users to specify custom stocks and view predictions with accuracy metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import our modules
from src.custom_stock_loader import CustomStockDataLoader
from src.model_plstm_tal import PLSTM_TAL
from src.baselines import BaselineModelFactory
from src.cae import CAEFeatureExtractor
from src.train import DataPreprocessor, ModelTrainer
from src.eval import ModelEvaluator
from src.eemd import EEMDDenoiser

# Import Bayesian optimizer (optional)
try:
    from src.bayesian_optimizer import BayesianOptimizer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Import indicators (try TA-Lib first, fallback to manual)
try:
    from src.indicators_talib import TechnicalIndicatorsTA
    # Test if TA-Lib actually works
    TechnicalIndicatorsTA()
    TALIB_AVAILABLE = True
except (ImportError, Exception):
    TALIB_AVAILABLE = False

from src.indicators import TechnicalIndicators

# Set page config
st.set_page_config(
    page_title="PLSTM-TAL Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 PLSTM-TAL Stock Market Prediction</h1>
        <p>Enhanced prediction using Peephole LSTM with Temporal Attention Layer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Stock selection
    st.sidebar.subheader("📊 Stock Selection")
    
    # Load the custom stock loader
    @st.cache_resource
    def get_stock_loader():
        return CustomStockDataLoader()
    
    stock_loader = get_stock_loader()
    
    # Popular symbols for quick selection
    popular_symbols = stock_loader.get_popular_symbols()
    
    # Option to choose from popular symbols or enter custom
    selection_mode = st.sidebar.radio(
        "Selection Mode:",
        ["Popular Symbols", "Custom Symbol"]
    )
    
    if selection_mode == "Popular Symbols":
        category = st.sidebar.selectbox("Category:", list(popular_symbols.keys()))
        symbol = st.sidebar.selectbox("Symbol:", popular_symbols[category])
    else:
        symbol = st.sidebar.text_input(
            "Stock Symbol:", 
            value="AAPL",
            help="Enter any valid Yahoo Finance symbol (e.g., AAPL, TSLA, ^GSPC, RELIANCE.NS)"
        ).strip().upper()
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    # Preset date ranges
    date_preset = st.sidebar.selectbox(
        "Preset Range:",
        ["Last 10 Years", "Last 5 Years", "Last 3 Years", "Last 2 Years", "Last 1 Year", "Paper Default (2005-2022)", "Custom"]
    )
    
    if date_preset == "Paper Default (2005-2022)":
        start_date = "2005-01-01"
        end_date = "2022-03-31"
    elif date_preset == "Last 10 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=10*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 5 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 3 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 2 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 1 Year":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1)).strftime("%Y-%m-%d")
        with col2:
            end_date = st.date_input("End Date", value=datetime.now()).strftime("%Y-%m-%d")
    
    # Model selection
    st.sidebar.subheader("🎯 Model Selection")
    available_models = ["LSTM", "PLSTM-TAL", "CNN", "SVM", "Random Forest"]
    selected_models = st.sidebar.multiselect(
        "Select Models to Train and Compare:", 
        available_models,
        default=["LSTM"]  # Default to LSTM as requested
    )
    
    if not selected_models:
        st.sidebar.warning("Please select at least one model.")
        return
    
    # Model configuration (keeping paper-compliant defaults)
    st.sidebar.subheader("🤖 Model Configuration")
    
    use_paper_params = st.sidebar.checkbox("Use Paper-Compliant Parameters", value=True)
    
    if use_paper_params:
        # Paper-compliant parameters (fixed)
        model_config = {
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1,
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'window_length': 20,
            'epochs': 100,
            'optimizer': 'adamax'
        }
        st.sidebar.info("Using paper-compliant PLSTM-TAL parameters")
    else:
        # Allow some customization
        model_config = {
            'hidden_size': st.sidebar.slider("Hidden Size", 32, 128, 64),
            'num_layers': st.sidebar.slider("Num Layers", 1, 3, 1),
            'dropout': st.sidebar.slider("Dropout", 0.0, 0.5, 0.1),
            'window_length': st.sidebar.slider("Window Length", 10, 50, 20),
            'epochs': st.sidebar.slider("Training Epochs", 50, 200, 100),
            'batch_size': st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1),
            'learning_rate': st.sidebar.selectbox("Learning Rate", [1e-4, 1e-3, 1e-2], index=1),
            'optimizer': st.sidebar.selectbox("Optimizer", ['adam', 'adamax', 'rmsprop'], index=1),
            'activation': 'tanh'
        }
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        use_bayesian = st.checkbox("Enable Bayesian Optimization", value=False) if BAYESIAN_AVAILABLE else False
        if not BAYESIAN_AVAILABLE:
            st.warning("⚠️ Bayesian optimization not available (scikit-optimize not installed)")
        use_quick_mode = st.checkbox("Quick Mode (Faster Training)", value=True)
        save_model = st.checkbox("Save Trained Model", value=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock Data", "🤖 Model Training", "📊 Predictions", "📋 Results"])
    
    with tab1:
        st.header("Stock Data Analysis")
        
        if st.button("🔄 Load Stock Data", type="primary"):
            with st.spinner(f"Loading data for {symbol}..."):
                try:
                    # Validate symbol first
                    is_valid, message = stock_loader.validate_symbol(symbol)
                    
                    if not is_valid:
                        st.error(f"❌ {message}")
                        return
                    
                    st.success(f"✅ {message}")
                    
                    # Get stock info
                    stock_info = stock_loader.get_stock_info(symbol)
                    
                    # Display stock information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Company", stock_info['name'])
                    with col2:
                        st.metric("Country", stock_info['country'])
                    with col3:
                        st.metric("Exchange", stock_info['exchange'])
                    
                    # Download stock data
                    stock_data = stock_loader.download_stock_data(symbol, start_date, end_date)
                    st.session_state.stock_data = stock_data
                    
                    # Display data summary
                    st.subheader("📊 Data Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Days", len(stock_data))
                    with col2:
                        st.metric("Date Range", f"{len(stock_data)} days")
                    with col3:
                        currency = stock_loader._detect_currency(symbol)
                        st.metric("Price Range", f"{currency}{stock_data['close'].min():.2f} - {currency}{stock_data['close'].max():.2f}")
                    with col4:
                        total_return = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1) * 100
                        st.metric("Total Return", f"{total_return:.2f}%")
                    
                    # Plot stock price
                    st.subheader("📈 Stock Price Chart")
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                    
                    # Price chart
                    ax1.plot(stock_data.index, stock_data['close'], linewidth=1.5, label='Close Price')
                    ax1.set_title(f'{symbol} Stock Price')
                    ax1.set_ylabel('Price')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    # Volume chart
                    ax2.bar(stock_data.index, stock_data['volume'], alpha=0.6, color='orange')
                    ax2.set_title('Trading Volume')
                    ax2.set_ylabel('Volume')
                    ax2.set_xlabel('Date')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display data table
                    st.subheader("📋 Recent Data")
                    st.dataframe(stock_data.tail(10))
                    
                except Exception as e:
                    st.error(f"❌ Error loading stock data: {str(e)}")
    
    with tab2:
        st.header("Model Training")
        
        if st.session_state.stock_data is None:
            st.warning("⚠️ Please load stock data first in the 'Stock Data' tab.")
            return
        
        if st.button("🚀 Train Selected Models", type="primary"):
            train_models(st.session_state.stock_data, symbol, selected_models, model_config, use_bayesian, use_quick_mode, save_model)
    
    with tab3:
        st.header("Predictions")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
            return
        
        display_predictions()
    
    with tab4:
        st.header("Detailed Results")
        
        if st.session_state.prediction_results is None:
            st.warning("⚠️ No results available. Please train the model first.")
            return
        
        display_detailed_results()

def train_model(stock_data, symbol, model_config, use_bayesian, use_quick_mode, save_model):
    """Train the PLSTM-TAL model with the given configuration."""
    
    try:
        with st.spinner("Training PLSTM-TAL model... This may take a few minutes."):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate technical indicators
            status_text.text("Step 1/6: Computing technical indicators...")
            progress_bar.progress(10)
            
            if TALIB_AVAILABLE:
                try:
                    indicators = TechnicalIndicatorsTA()
                    st.info("Using TA-Lib for paper-compliant indicators")
                except ImportError:
                    indicators = TechnicalIndicators()
                    st.warning("TA-Lib not available, using manual indicators")
            else:
                indicators = TechnicalIndicators()
                st.warning("Using manual indicators (TA-Lib not available)")
            
            features_df = indicators.compute_features(stock_data)
            
            # Step 2: EEMD filtering
            status_text.text("Step 2/6: Applying EEMD denoising...")
            progress_bar.progress(20)
            
            denoiser = EEMDDenoiser(n_ensembles=50, noise_scale=0.2, w=7)
            filtered_prices, eemd_metadata = denoiser.process_price_series(stock_data['close'])
            
            # Step 3: CAE feature extraction
            status_text.text("Step 3/6: Training Contractive Autoencoder...")
            progress_bar.progress(30)
            
            cae = CAEFeatureExtractor(
                hidden_dim=64,
                encoding_dim=16,
                dropout=0.1,
                lambda_reg=1e-4
            )
            
            cae_epochs = 50 if use_quick_mode else 100
            cae_history = cae.train(
                features_df, 
                epochs=cae_epochs,
                batch_size=32,
                learning_rate=1e-3,
                verbose=False
            )
            
            # Step 4: Data preparation
            status_text.text("Step 4/6: Preparing training data...")
            progress_bar.progress(40)
            
            preprocessor = DataPreprocessor(
                window_length=model_config['window_length'],
                step_size=1
            )
            
            X_sequences, y_labels = preprocessor.prepare_data(
                features_df, stock_data['close'], cae, filtered_prices
            )
            
            # Train/validation/test split
            from sklearn.model_selection import train_test_split
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_labels, test_size=0.15, random_state=42, stratify=y_labels
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
            )
            
            input_size = X_train.shape[2]
            
            # Step 5: Model training
            status_text.text("Step 5/6: Training PLSTM-TAL model...")
            progress_bar.progress(60)
            
            # Use Bayesian optimization if enabled
            if use_bayesian and BAYESIAN_AVAILABLE:
                try:
                    optimizer = BayesianOptimizer(X_train, y_train, X_val, y_val, input_size, n_calls=10)
                    optimized_params = optimizer.optimize(verbose=False)
                    
                    # Update model config with optimized parameters
                    model_config.update(optimized_params)
                    st.success("✅ Bayesian optimization completed")
                except Exception as e:
                    st.warning(f"⚠️ Bayesian optimization failed: {e}, using default parameters")
            
            # Create and train PLSTM-TAL model
            plstm_model = PLSTM_TAL(
                input_size=input_size,
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers'],
                dropout=model_config['dropout'],
                activation=model_config['activation']
            )
            
            plstm_trainer = ModelTrainer(plstm_model)
            training_epochs = model_config['epochs'] // 2 if use_quick_mode else model_config['epochs']
            
            plstm_history = plstm_trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=training_epochs,
                batch_size=model_config['batch_size'],
                learning_rate=model_config['learning_rate'],
                optimizer_name=model_config['optimizer'],
                early_stopping_patience=20
            )
            
            # Step 6: Model evaluation
            status_text.text("Step 6/6: Evaluating model performance...")
            progress_bar.progress(80)
            
            evaluator = ModelEvaluator(save_plots=False)
            
            # Evaluate PLSTM-TAL
            plstm_result = evaluator.evaluate_model(
                plstm_model, X_test, y_test, "PLSTM-TAL", is_torch_model=True
            )
            
            # Train and evaluate baselines for comparison
            baseline_results = {}
            
            # LSTM baseline
            lstm_model = BaselineModelFactory.create_model('lstm', input_size, hidden_size=32, dropout=0.1)
            lstm_trainer = ModelTrainer(lstm_model)
            lstm_trainer.train(X_train, y_train, X_val, y_val, epochs=training_epochs//2, batch_size=32, learning_rate=1e-3, optimizer_name='adamax')
            baseline_results['LSTM'] = evaluator.evaluate_model(lstm_model, X_test, y_test, "LSTM", is_torch_model=True)
            
            # SVM baseline
            svm_model = BaselineModelFactory.create_model('svm', input_size, C=1.0, gamma='scale')
            svm_model.fit(X_train, y_train)
            baseline_results['SVM'] = evaluator.evaluate_model(svm_model, X_test, y_test, "SVM", is_torch_model=False)
            
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            # Store results
            st.session_state.prediction_results = {
                'plstm_result': plstm_result,
                'baseline_results': baseline_results,
                'model_config': model_config,
                'data_info': {
                    'symbol': symbol,
                    'total_samples': len(stock_data),
                    'features_count': len(features_df.columns),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'models': {
                    'plstm': plstm_model,
                    'cae': cae
                }
            }
            st.session_state.model_trained = True
            
            # Save model if requested
            if save_model:
                torch.save(plstm_model.state_dict(), f'plstm_tal_{symbol}.pth')
                st.success(f"✅ Model saved as plstm_tal_{symbol}.pth")
            
            st.success("🎉 Model training completed successfully!")
            
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def train_models(stock_data, symbol, selected_models, model_config, use_bayesian, use_quick_mode, save_model):
    """Train multiple selected models with the given configuration."""
    
    try:
        results = {}
        all_training_times = {}
        
        with st.spinner(f"Training {len(selected_models)} selected models... This may take a few minutes."):
            
            # Progress tracking
            total_steps = 4 + len(selected_models)  # 4 preprocessing steps + training for each model
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate technical indicators
            status_text.text("Step 1/6: Computing technical indicators...")
            progress_bar.progress(10)
            
            if TALIB_AVAILABLE:
                try:
                    indicators = TechnicalIndicatorsTA()
                    st.info("Using TA-Lib for paper-compliant indicators")
                except ImportError:
                    indicators = TechnicalIndicators()
                    st.warning("TA-Lib not available, using manual indicators")
            else:
                indicators = TechnicalIndicators()
                st.warning("Using manual indicators (TA-Lib not available)")
            
            features_df = indicators.compute_features(stock_data)
            
            # Step 2: EEMD filtering
            status_text.text("Step 2/6: Applying EEMD denoising...")
            progress_bar.progress(20)
            
            denoiser = EEMDDenoiser(n_ensembles=50, noise_scale=0.2, w=7)
            filtered_prices, eemd_metadata = denoiser.process_price_series(stock_data['close'])
            
            # Step 3: CAE feature extraction
            status_text.text("Step 3/6: Training Contractive Autoencoder...")
            progress_bar.progress(30)
            
            cae = CAEFeatureExtractor(
                hidden_dim=64,
                encoding_dim=16,
                dropout=0.1,
                lambda_reg=1e-4
            )
            
            cae_epochs = 50 if use_quick_mode else 100
            cae_history = cae.train(
                features_df, 
                epochs=cae_epochs,
                batch_size=32,
                learning_rate=1e-3,
                verbose=False
            )
            
            # Step 4: Data preparation
            status_text.text("Step 4/6: Preparing training data...")
            progress_bar.progress(40)
            
            preprocessor = DataPreprocessor(
                window_length=model_config['window_length'],
                step_size=1
            )
            
            X_sequences, y_labels = preprocessor.prepare_data(
                features_df, stock_data['close'], cae, filtered_prices
            )
            
            # Train/validation/test split
            from sklearn.model_selection import train_test_split
            
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_sequences, y_labels, test_size=0.15, random_state=42, stratify=y_labels
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
            )
            
            input_size = X_train.shape[2]
            
            # Train each selected model
            for i, model_name in enumerate(selected_models):
                step_num = 5 + i
                status_text.text(f"Step {step_num}/{total_steps}: Training {model_name} model...")
                progress_bar.progress(50 + (i * 40 // len(selected_models)))
                
                start_time = datetime.now()
                
                if model_name == "PLSTM-TAL":
                    # Use Bayesian optimization if enabled
                    if use_bayesian and BAYESIAN_AVAILABLE:
                        try:
                            optimizer = BayesianOptimizer(X_train, y_train, X_val, y_val, input_size, n_calls=10)
                            optimized_params = optimizer.optimize(verbose=False)
                            
                            # Update model config with optimized parameters
                            model_config.update(optimized_params)
                            st.success("✅ Bayesian optimization completed")
                        except Exception as e:
                            st.warning(f"⚠️ Bayesian optimization failed: {e}, using default parameters")
                    
                    # Create and train PLSTM-TAL model
                    plstm_model = PLSTM_TAL(
                        input_size=input_size,
                        hidden_size=model_config['hidden_size'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout'],
                        activation=model_config['activation']
                    )
                    
                    plstm_trainer = ModelTrainer(plstm_model)
                    training_epochs = model_config['epochs'] // 2 if use_quick_mode else model_config['epochs']
                    
                    plstm_history = plstm_trainer.train(
                        X_train, y_train, X_val, y_val,
                        epochs=training_epochs,
                        batch_size=model_config['batch_size'],
                        learning_rate=model_config['learning_rate'],
                        optimizer_name=model_config['optimizer'],
                        early_stopping_patience=20
                    )
                    
                    # Evaluate PLSTM-TAL model
                    plstm_evaluator = ModelEvaluator()
                    plstm_result = plstm_evaluator.evaluate_model(
                        plstm_model, X_test, y_test, "PLSTM-TAL", is_torch_model=True
                    )
                    
                    results['plstm_result'] = {
                        'model': plstm_model,
                        'result': plstm_result,
                        'history': plstm_history
                    }
                    
                    # Save model if requested
                    if save_model:
                        model_filename = f"plstm_tal_{symbol.replace('.', '_')}.pth"
                        torch.save(plstm_model.state_dict(), model_filename)
                        st.success(f"✅ PLSTM-TAL model saved as {model_filename}")
                        
                elif model_name == "LSTM":
                    # Create and train LSTM baseline
                    lstm_model = BaselineModelFactory.create_model(
                        'lstm', 
                        input_size=input_size,
                        hidden_size=model_config['hidden_size'],
                        num_layers=model_config['num_layers'],
                        dropout=model_config['dropout']
                    )
                    
                    lstm_trainer = ModelTrainer(lstm_model)
                    training_epochs = model_config['epochs'] // 2 if use_quick_mode else model_config['epochs']
                    
                    lstm_history = lstm_trainer.train(
                        X_train, y_train, X_val, y_val,
                        epochs=training_epochs,
                        batch_size=model_config['batch_size'],
                        learning_rate=model_config['learning_rate'],
                        optimizer_name=model_config['optimizer'],
                        early_stopping_patience=20
                    )
                    
                    # Evaluate LSTM model
                    lstm_evaluator = ModelEvaluator()
                    lstm_result = lstm_evaluator.evaluate_model(
                        lstm_model, X_test, y_test, "LSTM", is_torch_model=True
                    )
                    
                    results['lstm_result'] = {
                        'model': lstm_model,
                        'result': lstm_result,
                        'history': lstm_history
                    }
                    
                    # Save model if requested
                    if save_model:
                        model_filename = f"lstm_{symbol.replace('.', '_')}.pth"
                        torch.save(lstm_model.state_dict(), model_filename)
                        st.success(f"✅ LSTM model saved as {model_filename}")
                        
                elif model_name == "CNN":
                    # Create and train CNN baseline
                    cnn_model = BaselineModelFactory.create_model(
                        'cnn', 
                        input_size=input_size,
                        seq_len=model_config['window_length']
                    )
                    
                    cnn_trainer = ModelTrainer(cnn_model)
                    training_epochs = model_config['epochs'] // 2 if use_quick_mode else model_config['epochs']
                    
                    cnn_history = cnn_trainer.train(
                        X_train, y_train, X_val, y_val,
                        epochs=training_epochs,
                        batch_size=model_config['batch_size'],
                        learning_rate=model_config['learning_rate'],
                        optimizer_name=model_config['optimizer'],
                        early_stopping_patience=20
                    )
                    
                    # Evaluate CNN model
                    cnn_evaluator = ModelEvaluator()
                    cnn_result = cnn_evaluator.evaluate_model(
                        cnn_model, X_test, y_test, "CNN", is_torch_model=True
                    )
                    
                    results['cnn_result'] = {
                        'model': cnn_model,
                        'result': cnn_result,
                        'history': cnn_history
                    }
                    
                    # Save model if requested
                    if save_model:
                        model_filename = f"cnn_{symbol.replace('.', '_')}.pth"
                        torch.save(cnn_model.state_dict(), model_filename)
                        st.success(f"✅ CNN model saved as {model_filename}")
                        
                elif model_name == "SVM":
                    # Create and train SVM baseline
                    svm_model = BaselineModelFactory.create_model('svm', input_size=input_size)
                    
                    # Convert to numpy for SVM
                    X_train_np = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
                    y_train_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
                    X_test_np = X_test.numpy() if hasattr(X_test, 'numpy') else X_test
                    y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else y_test
                    
                    svm_model.fit(X_train_np, y_train_np)
                    
                    # Evaluate SVM model
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    svm_pred = svm_model.predict(X_test_np)
                    svm_proba = svm_model.predict_proba(X_test_np)
                    
                    svm_result = {
                        'test_accuracy': accuracy_score(y_test_np, svm_pred),
                        'precision': precision_score(y_test_np, svm_pred, average='binary'),
                        'recall': recall_score(y_test_np, svm_pred, average='binary'),
                        'f1_score': f1_score(y_test_np, svm_pred, average='binary'),
                        'predictions': svm_pred,
                        'probabilities': svm_proba
                    }
                    
                    results['svm_result'] = {
                        'model': svm_model,
                        'result': svm_result
                    }
                    
                elif model_name == "Random Forest":
                    # Create and train Random Forest baseline
                    rf_model = BaselineModelFactory.create_model('rf', input_size=input_size)
                    
                    # Convert to numpy for RF
                    X_train_np = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
                    y_train_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
                    X_test_np = X_test.numpy() if hasattr(X_test, 'numpy') else X_test
                    y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else y_test
                    
                    rf_model.fit(X_train_np, y_train_np)
                    
                    # Evaluate RF model
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    rf_pred = rf_model.predict(X_test_np)
                    rf_proba = rf_model.predict_proba(X_test_np)
                    
                    rf_result = {
                        'test_accuracy': accuracy_score(y_test_np, rf_pred),
                        'precision': precision_score(y_test_np, rf_pred, average='binary'),
                        'recall': recall_score(y_test_np, rf_pred, average='binary'),
                        'f1_score': f1_score(y_test_np, rf_pred, average='binary'),
                        'predictions': rf_pred,
                        'probabilities': rf_proba
                    }
                    
                    results['rf_result'] = {
                        'model': rf_model,
                        'result': rf_result
                    }
                
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()
                all_training_times[model_name] = training_time
                st.success(f"✅ {model_name} training completed in {training_time:.1f}s")
            
            # Final step
            status_text.text("Step 6/6: Finalizing results...")
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state.prediction_results = {
                'selected_models': selected_models,
                'results': results,
                'training_times': all_training_times,
                'symbol': symbol,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': input_size,
                'model_config': model_config
            }
            st.session_state.model_trained = True
            
            status_text.text("✅ All models trained successfully!")
            st.success(f"🎉 Successfully trained {len(selected_models)} models!")
            
            # Display quick summary
            st.subheader("🏆 Quick Results Summary")
            for model_name in selected_models:
                if model_name.lower().replace('-', '_').replace(' ', '_') + '_result' in results:
                    result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
                    result = results[result_key]['result']
                    
                    # Extract accuracy from the correct structure
                    if isinstance(result, dict) and 'metrics' in result:
                        accuracy = result['metrics'].get('accuracy', 0)
                    elif isinstance(result, dict):
                        accuracy = result.get('test_accuracy', 0)
                    else:
                        accuracy = result.test_accuracy if hasattr(result, 'test_accuracy') else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{model_name} Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Training Time", f"{all_training_times[model_name]:.1f}s")
                    with col3:
                        st.metric("Status", "✅ Complete")
                    
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

def display_predictions():
    """Display model predictions and accuracy metrics for selected models."""
    
    results = st.session_state.prediction_results
    selected_models = results['selected_models']
    model_results = results['results']
    training_times = results['training_times']
    
    st.subheader("🎯 Model Performance Comparison")
    
    # Performance metrics table
    metrics_data = []
    
    for model_name in selected_models:
        result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
        
        if result_key in model_results:
            model_info = model_results[result_key]
            result = model_info['result']
            
            # Extract metrics from the result structure
            if isinstance(result, dict) and 'metrics' in result:
                # For models evaluated with ModelEvaluator
                metrics = result['metrics']
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}",
                    'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
                    'Training Time (s)': f"{training_times.get(model_name, 0):.1f}"
                })
            elif isinstance(result, dict):
                # For traditional ML models with direct metric storage
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result.get('test_accuracy', 0):.3f}",
                    'Precision': f"{result.get('precision', 0):.3f}",
                    'Recall': f"{result.get('recall', 0):.3f}",
                    'F1-Score': f"{result.get('f1_score', 0):.3f}",
                    'Training Time (s)': f"{training_times.get(model_name, 0):.1f}"
                })
            else:
                # Legacy support for object-style results
                accuracy = result.test_accuracy if hasattr(result, 'test_accuracy') else 0
                precision = result.precision if hasattr(result, 'precision') else 0
                recall = result.recall if hasattr(result, 'recall') else 0
                f1 = result.f1_score if hasattr(result, 'f1_score') else 0
                
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{accuracy:.3f}",
                    'Precision': f"{precision:.3f}",
                    'Recall': f"{recall:.3f}",
                    'F1-Score': f"{f1:.3f}",
                    'Training Time (s)': f"{training_times.get(model_name, 0):.1f}"
                })
    
    # Display metrics table
    import pandas as pd
    comparison_df = pd.DataFrame(metrics_data)
    
    if not comparison_df.empty:
        st.dataframe(comparison_df.set_index('Model'), use_container_width=True)
        
        # Individual model metrics
        cols = st.columns(len(selected_models))
        
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
                
                if result_key in model_results:
                    model_info = model_results[result_key]
                    result = model_info['result']
                    
                    # Extract accuracy from the correct structure
                    if isinstance(result, dict) and 'metrics' in result:
                        accuracy = result['metrics'].get('accuracy', 0)
                    elif isinstance(result, dict):
                        accuracy = result.get('test_accuracy', 0)
                    else:
                        accuracy = result.test_accuracy if hasattr(result, 'test_accuracy') else 0
                    
                    st.metric(
                        f"{model_name}",
                        f"{accuracy:.1%}",
                        delta=f"{training_times.get(model_name, 0):.1f}s"
                    )
        
        # Plot performance comparison
        st.subheader("📊 Performance Visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        models = [row['Model'] for row in metrics_data]
        accuracies = [float(row['Accuracy']) for row in metrics_data]
        
        ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        times = [float(row['Training Time (s)']) for row in metrics_data]
        
        ax2.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
        ax2.set_title('Training Time Comparison')
        ax2.set_ylabel('Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(times):
            ax2.text(i, v + max(times) * 0.01, f'{v:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Best model highlight
        best_model_idx = accuracies.index(max(accuracies))
        best_model = models[best_model_idx]
        best_accuracy = accuracies[best_model_idx]
        
        st.success(f"🏆 Best performing model: **{best_model}** with {best_accuracy:.1%} accuracy")
        
    else:
        st.warning("No model results available to display.")

def display_detailed_results():
    """Display detailed results and analysis."""
    
    results = st.session_state.prediction_results
    selected_models = results['selected_models']
    model_results = results['results']
    model_config = results['model_config']
    
    # Data information
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"""
        **Stock Symbol:** {results['symbol']}
        **Training Samples:** {results['train_samples']}
        **Test Samples:** {results['test_samples']}
        """)
    with col2:
        st.info(f"""
        **Feature Count:** {results['feature_count']}
        **Window Length:** {model_config['window_length']}
        **Selected Models:** {len(selected_models)}
        """)
    with col3:
        st.info(f"""
        **Total Training Time:** {sum(results['training_times'].values()):.1f}s
        **Average Time/Model:** {sum(results['training_times'].values())/len(selected_models):.1f}s
        """)
    
    # Model configuration details
    st.subheader("🔧 Model Configuration")
    config_df = pd.DataFrame([
        {'Parameter': k.replace('_', ' ').title(), 'Value': str(v)} 
        for k, v in model_config.items()
    ])
    st.dataframe(config_df.set_index('Parameter'), use_container_width=True)
    
    # Individual model details
    st.subheader("🤖 Individual Model Results")
    
    for model_name in selected_models:
        result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
        
        if result_key in model_results:
            with st.expander(f"📈 {model_name} Details"):
                model_info = model_results[result_key]
                result = model_info['result']
                training_time = results['training_times'].get(model_name, 0)
                
                # Display metrics
                if isinstance(result, dict) and 'metrics' in result:
                    # For models evaluated with ModelEvaluator
                    metrics = result['metrics']
                    metrics_dict = {
                        'Test Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                        'Precision': f"{metrics.get('precision', 0):.3f}",
                        'Recall': f"{metrics.get('recall', 0):.3f}",
                        'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
                        'Training Time': f"{training_time:.1f}s"
                    }
                elif isinstance(result, dict):
                    # For traditional ML models with direct metric storage
                    metrics_dict = {
                        'Test Accuracy': f"{result.get('test_accuracy', 0):.3f}",
                        'Precision': f"{result.get('precision', 0):.3f}",
                        'Recall': f"{result.get('recall', 0):.3f}",
                        'F1-Score': f"{result.get('f1_score', 0):.3f}",
                        'Training Time': f"{training_time:.1f}s"
                    }
                else:
                    # Legacy support for object-style results
                    metrics_dict = {
                        'Test Accuracy': f"{result.test_accuracy:.3f}" if hasattr(result, 'test_accuracy') else "N/A",
                        'Precision': f"{result.precision:.3f}" if hasattr(result, 'precision') else "N/A",
                        'Recall': f"{result.recall:.3f}" if hasattr(result, 'recall') else "N/A",
                        'F1-Score': f"{result.f1_score:.3f}" if hasattr(result, 'f1_score') else "N/A",
                        'Training Time': f"{training_time:.1f}s"
                    }
                
                metric_cols = st.columns(len(metrics_dict))
                for i, (metric, value) in enumerate(metrics_dict.items()):
                    with metric_cols[i]:
                        st.metric(metric, value)
                
                # Show training history for neural network models
                if 'history' in model_info and model_info['history'] is not None:
                    st.write("**Training History:**")
                    history = model_info['history']
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Plot training loss
                    if 'train_losses' in history:
                        ax1.plot(history['train_losses'], label='Training Loss')
                        if 'val_losses' in history:
                            ax1.plot(history['val_losses'], label='Validation Loss')
                        ax1.set_title('Training Loss')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Loss')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                    
                    # Plot training accuracy
                    if 'train_accuracies' in history:
                        ax2.plot(history['train_accuracies'], label='Training Accuracy')
                        if 'val_accuracies' in history:
                            ax2.plot(history['val_accuracies'], label='Validation Accuracy')
                        ax2.set_title('Training Accuracy')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Accuracy')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

    # Download results
    st.subheader("💾 Download Results")
    
    if st.button("📥 Download Results as JSON"):
        # Prepare results for download
        download_results = {
            'metadata': {
                'symbol': results['symbol'],
                'train_samples': results['train_samples'],
                'test_samples': results['test_samples'],
                'feature_count': results['feature_count'],
                'selected_models': selected_models,
                'timestamp': datetime.now().isoformat()
            },
            'model_config': model_config,
            'training_times': results['training_times'],
            'model_performance': {}
        }
        
        # Add performance for each model
        for model_name in selected_models:
            result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
            if result_key in model_results:
                model_info = model_results[result_key]
                result = model_info['result']
                
                if isinstance(result, dict) and 'metrics' in result:
                    # For models evaluated with ModelEvaluator
                    metrics = result['metrics']
                    download_results['model_performance'][model_name] = {
                        'test_accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0),
                        'auc_roc': metrics.get('auc_roc', 0),
                        'pr_auc': metrics.get('pr_auc', 0),
                        'mcc': metrics.get('mcc', 0)
                    }
                elif isinstance(result, dict):
                    # For traditional ML models with direct metric storage
                    download_results['model_performance'][model_name] = result
                else:
                    # Legacy support for object-style results
                    download_results['model_performance'][model_name] = {
                        'test_accuracy': result.test_accuracy if hasattr(result, 'test_accuracy') else 0,
                        'precision': result.precision if hasattr(result, 'precision') else 0,
                        'recall': result.recall if hasattr(result, 'recall') else 0,
                        'f1_score': result.f1_score if hasattr(result, 'f1_score') else 0
                    }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(download_results, indent=2),
            file_name=f"multi_model_results_{results['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()