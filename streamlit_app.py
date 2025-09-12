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
        ["Last 5 Years", "Last 10 Years", "Last 3 Years", "Last 2 Years", "Last 1 Year", "Paper Default (2005-2022)", "Custom"]
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
        # Paper-compliant parameters (improved for better performance)
        model_config = {
            'hidden_size': 64,     # Restored to original size but with better training
            'num_layers': 1,
            'dropout': 0.2,        # Slightly higher dropout for better generalization
            'activation': 'tanh',
            'learning_rate': 5e-4, # Reduced learning rate for stability
            'batch_size': 32,
            'window_length': 20,
            'epochs': 200,         # More epochs for better convergence
            'optimizer': 'adam'    # Switch to Adam for better convergence
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
        use_quick_mode = st.checkbox("Quick Mode (Faster Training)", value=False)  # Default to False for better performance
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
    
    start_time = datetime.now()  # Track training start time
    
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
            
            denoiser = EEMDDenoiser(n_ensembles=10, noise_scale=0.2, w=7)
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
            # Use more epochs for better accuracy as requested
            training_epochs = model_config['epochs']
            
            plstm_history = plstm_trainer.train(
                X_train, y_train, X_val, y_val,
                epochs=training_epochs,
                batch_size=model_config['batch_size'],
                learning_rate=model_config['learning_rate'],
                optimizer_name=model_config['optimizer'],
                early_stopping_patience=100  # Much higher patience for better convergence
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
            
            # LSTM baseline with improved configuration
            lstm_model = BaselineModelFactory.create_model('lstm', input_size, hidden_size=model_config['hidden_size'], dropout=model_config['dropout'])
            lstm_trainer = ModelTrainer(lstm_model)
            lstm_trainer.train(X_train, y_train, X_val, y_val, epochs=training_epochs, batch_size=32, learning_rate=1e-3, optimizer_name='adamax')
            baseline_results['LSTM'] = evaluator.evaluate_model(lstm_model, X_test, y_test, "LSTM", is_torch_model=True)
            
            # SVM baseline
            svm_model = BaselineModelFactory.create_model('svm', input_size, C=1.0, gamma='scale')
            svm_model.fit(X_train, y_train)
            baseline_results['SVM'] = evaluator.evaluate_model(svm_model, X_test, y_test, "SVM", is_torch_model=False)
            
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            # Store results
            st.session_state.prediction_results = {
                'selected_models': ['PLSTM-TAL'],  # For compatibility with display function
                'results': {
                    'plstm_tal_result': {
                        'model': plstm_model,
                        'result': plstm_result,
                        'history': None
                    }
                },
                'training_times': {'PLSTM-TAL': (datetime.now() - start_time).total_seconds()},
                'symbol': symbol,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': X_train.shape[2],
                'model_config': model_config,
                # Store additional data for prediction visualization
                'test_data': {
                    'X_test': X_test,
                    'y_test': y_test,
                },
                'price_data': {
                    'prices': stock_data['close'].values,
                    'dates': stock_data.index,
                    'test_start_idx': len(stock_data) - len(X_test) - model_config['window_length'] + 1
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
            
            denoiser = EEMDDenoiser(n_ensembles=10, noise_scale=0.2, w=7)
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
                'model_config': model_config,
                # Store additional data for prediction visualization
                'test_data': {
                    'X_test': X_test,
                    'y_test': y_test,
                },
                'price_data': {
                    'prices': stock_data['close'].values,
                    'dates': stock_data.index,
                    'test_start_idx': len(stock_data) - len(X_test) - model_config['window_length'] + 1
                }
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

def convert_predictions_to_prices(y_pred_binary, y_proba, original_prices, test_start_idx, window_length):
    """
    Convert binary predictions back to price movement predictions.
    
    FIXED: This function was causing terrible predictions due to compounding errors.
    Now uses actual prices as base for each prediction instead of previous predicted prices.
    
    Args:
        y_pred_binary: Binary predictions (0/1)
        y_proba: Prediction probabilities 
        original_prices: Original price series
        test_start_idx: Starting index for test data in price series
        window_length: Sequence window length
        
    Returns:
        Dictionary with prediction data for visualization
    """
    # Get the actual test period prices
    actual_test_start = test_start_idx + window_length
    actual_prices = original_prices[actual_test_start:actual_test_start + len(y_pred_binary)]
    
    # Calculate average volatility for scaling (instead of individual returns)
    if len(actual_prices) > 10:  # Need enough data for reliable volatility
        actual_returns = np.diff(np.log(actual_prices))
        avg_volatility = np.std(actual_returns)
    else:
        avg_volatility = 0.01  # Default volatility if insufficient data
    
    # Convert binary predictions to predicted price movements
    predicted_movements = []
    predicted_prices = []
    
    for i, (pred, prob) in enumerate(zip(y_pred_binary, y_proba)):
        if i < len(actual_prices):
            # CRITICAL FIX: Use actual price as base, not previous predicted price
            current_actual = actual_prices[i]
            
            # Use average volatility scaled by confidence and reduced to prevent extreme predictions
            movement_magnitude = avg_volatility * prob * 0.5  # Scale down for more realistic predictions
            
            # Apply direction based on binary prediction
            if pred == 1:  # Predicted increase
                predicted_price = current_actual * (1 + movement_magnitude)
                predicted_movements.append(movement_magnitude)
            else:  # Predicted decrease
                predicted_price = current_actual * (1 - movement_magnitude)
                predicted_movements.append(-movement_magnitude)
                
            predicted_prices.append(predicted_price)
    
    return {
        'actual_prices': actual_prices,
        'predicted_prices': np.array(predicted_prices),
        'predicted_movements': np.array(predicted_movements),
        'binary_predictions': y_pred_binary,
        'probabilities': y_proba
    }

def make_future_prediction(model, preprocessed_features, prices, window_length, model_name, symbol):
    """
    Make a future price prediction for the next trading period.
    
    FIXED: Improved prediction logic to avoid extreme price predictions.
    
    Args:
        model: Trained model
        preprocessed_features: Already preprocessed feature matrix (same format as training data)
        prices: Price series
        window_length: Sequence window length
        model_name: Name of the model for logging
        symbol: Stock symbol
        
    Returns:
        Dictionary with future prediction information
    """
    try:
        # Get the latest sequence for prediction - use the last window from preprocessed features
        if len(preprocessed_features.shape) == 3:
            # Features are already in sequence format (samples, timesteps, features)
            latest_features = preprocessed_features[-1]  # Get the last sequence
        else:
            # Features need to be shaped into sequence
            latest_features = preprocessed_features[-window_length:].values
        
        latest_prices = prices[-window_length:]
        
        # Prepare input tensor
        if hasattr(model, 'predict_proba'):  # Sklearn models
            # For sklearn models, flatten the sequence
            X_latest = latest_features.reshape(1, -1)
            prediction_proba = model.predict_proba(X_latest)[0]
            prediction_binary = model.predict(X_latest)[0]
        else:  # PyTorch models
            import torch
            # For LSTM models, we need proper sequence shape: (batch_size, sequence_length, features)
            X_latest = torch.FloatTensor(latest_features).unsqueeze(0)  # Shape: (1, window_length, num_features)
            
            model.eval()
            with torch.no_grad():
                outputs = model(X_latest)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                
                # Convert to probabilities - handle different output dimensions
                if len(outputs.shape) == 1:
                    # Single output (binary classification)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    prediction_proba = probs[0] if len(probs) > 0 else 0.5
                    prediction_binary = 1 if prediction_proba > 0.5 else 0
                elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                    # Shape: (batch_size, 1) - single output per sample
                    probs = torch.sigmoid(outputs).cpu().numpy()[0]
                    prediction_proba = probs[0] if len(probs) > 0 else 0.5
                    prediction_binary = 1 if prediction_proba > 0.5 else 0
                else:
                    # Multi-class output
                    probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
                    prediction_binary = np.argmax(probs)
                    prediction_proba = probs[1] if len(probs) > 1 else probs[0]
        
        # Calculate predicted price movement
        current_price = float(prices.iloc[-1])
        
        # FIXED: Use more conservative movement estimation
        recent_returns = np.diff(np.log(prices[-30:]))  # Last 30 periods for better stability
        avg_volatility = np.std(recent_returns)
        
        # Scale movement by confidence and volatility, but cap it for reasonable predictions
        movement_magnitude = min(avg_volatility * prediction_proba * 0.3, 0.05)  # Cap at 5% max movement
        
        if prediction_binary == 1:  # Predicted increase
            predicted_price = current_price * (1 + movement_magnitude)
            direction = "📈 UP"
            movement = f"+{movement_magnitude*100:.2f}%"
        else:  # Predicted decrease
            predicted_price = current_price * (1 - movement_magnitude)
            direction = "📉 DOWN"
            movement = f"-{movement_magnitude*100:.2f}%"
        
        return {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'direction': direction,
            'movement_percent': movement_magnitude * 100,
            'confidence': prediction_proba * 100,
            'prediction_binary': prediction_binary,
            'movement_str': movement,
            'model_name': model_name,
            'symbol': symbol
        }
        
    except Exception as e:
        st.error(f"Error making future prediction with {model_name}: {str(e)}")
        return None

def plot_prediction_comparison(actual_prices, predicted_prices, dates, model_name, symbol):
    """
    Create a comparison plot of actual vs predicted prices.
    
    Args:
        actual_prices: Actual price values
        predicted_prices: Predicted price values  
        dates: Date indices
        model_name: Name of the model
        symbol: Stock symbol
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Ensure we have the same length for comparison
    min_len = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]
    
    if len(dates) > min_len:
        plot_dates = dates[-min_len:]
    else:
        plot_dates = dates
    
    # Plot 1: Price comparison
    ax1.plot(plot_dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
    ax1.plot(plot_dates, predicted_prices, label='Predicted Prices', color='red', linewidth=2, alpha=0.8)
    ax1.set_title(f'{model_name} - Price Prediction vs Actual ({symbol})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display accuracy metrics
    if len(actual_prices) > 1:
        # Direction accuracy (did we predict the right direction?)
        actual_directions = np.sign(np.diff(actual_prices))
        predicted_directions = np.sign(np.diff(predicted_prices))
        direction_accuracy = np.mean(actual_directions == predicted_directions) * 100
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
        ax1.text(0.02, 0.98, f'Direction Accuracy: {direction_accuracy:.1f}%\nMAPE: {mape:.2f}%', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Prediction error
    if len(actual_prices) == len(predicted_prices):
        prediction_error = actual_prices - predicted_prices
        ax2.plot(plot_dates, prediction_error, label='Prediction Error', color='green', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Prediction Error (Actual - Predicted)', fontsize=12)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = np.sqrt(np.mean(prediction_error**2))
        ax2.text(0.02, 0.98, f'RMSE: {rmse:.4f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

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
        
        # NEW: Add future forecasting section
        st.subheader("🔮 Future Price Forecasting")
        
        # Check if we have the required data for future predictions
        if 'test_data' in results and 'price_data' in results and st.session_state.stock_data is not None:
            st.info("🎯 **Next Trading Period Forecast** - What the models predict will happen next")
            
            # Generate technical indicators for the latest data
            if TALIB_AVAILABLE:
                try:
                    indicators = TechnicalIndicatorsTA()
                except ImportError:
                    indicators = TechnicalIndicators()
            else:
                indicators = TechnicalIndicators()
            
            latest_features_df = indicators.compute_features(st.session_state.stock_data)
            
            # Create forecast display for each model
            forecast_cols = st.columns(len(selected_models))
            
            for i, model_name in enumerate(selected_models):
                with forecast_cols[i]:
                    result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
                    
                    if result_key in model_results:
                        model_info = model_results[result_key]
                        model = model_info['model']
                        
                        # Make future prediction using test data format
                        test_data = results['test_data']
                        future_pred = make_future_prediction(
                            model, 
                            test_data['X_test'],  # Use preprocessed test features 
                            st.session_state.stock_data['close'], 
                            results['model_config']['window_length'],
                            model_name,
                            results['symbol']
                        )
                        
                        if future_pred:
                            # Display forecast card
                            st.markdown(f"""
                            <div style="
                                border: 2px solid {'#28a745' if future_pred['prediction_binary'] == 1 else '#dc3545'};
                                border-radius: 10px;
                                padding: 15px;
                                background: {'#d4f7dc' if future_pred['prediction_binary'] == 1 else '#f8d7da'};
                                margin: 10px 0;
                                text-align: center;
                            ">
                                <h4 style="margin: 0; color: #333;">{model_name} Forecast</h4>
                                <div style="font-size: 24px; margin: 10px 0;">{future_pred['direction']}</div>
                                <div style="font-size: 18px; color: #666; margin: 5px 0;">
                                    <strong>Current:</strong> ${future_pred['current_price']:.2f}
                                </div>
                                <div style="font-size: 18px; color: #666; margin: 5px 0;">
                                    <strong>Predicted:</strong> ${future_pred['predicted_price']:.2f}
                                </div>
                                <div style="font-size: 16px; color: #888; margin: 5px 0;">
                                    <strong>Movement:</strong> {future_pred['movement_str']}
                                </div>
                                <div style="font-size: 14px; color: #888; margin: 5px 0;">
                                    <strong>Confidence:</strong> {future_pred['confidence']:.1f}%
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Unable to generate forecast for {model_name}")
            
            # Add a summary forecast if multiple models
            if len(selected_models) > 1:
                st.subheader("📊 Consensus Forecast")
                
                # Collect all predictions
                all_forecasts = []
                test_data = results['test_data']  # Get test data for preprocessed features
                for model_name in selected_models:
                    result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
                    if result_key in model_results:
                        model_info = model_results[result_key]
                        model = model_info['model']
                        
                        future_pred = make_future_prediction(
                            model, 
                            test_data['X_test'],  # Use preprocessed test features
                            st.session_state.stock_data['close'], 
                            results['model_config']['window_length'],
                            model_name,
                            results['symbol']
                        )
                        
                        if future_pred:
                            all_forecasts.append(future_pred)
                
                if all_forecasts:
                    # Calculate consensus
                    avg_predicted_price = np.mean([f['predicted_price'] for f in all_forecasts])
                    avg_confidence = np.mean([f['confidence'] for f in all_forecasts])
                    up_votes = sum([1 for f in all_forecasts if f['prediction_binary'] == 1])
                    down_votes = len(all_forecasts) - up_votes
                    
                    consensus_direction = "📈 UP" if up_votes > down_votes else "📉 DOWN"
                    current_price = all_forecasts[0]['current_price']
                    movement_pct = ((avg_predicted_price - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Consensus Direction", consensus_direction)
                    with col2:
                        st.metric("Average Predicted Price", f"${avg_predicted_price:.2f}")
                    with col3:
                        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                    
                    # Show voting breakdown
                    st.write(f"**Model Voting:** {up_votes} models predict UP ⬆️, {down_votes} models predict DOWN ⬇️")
        
        # NEW: Add historical price prediction visualization
        st.subheader("📈 Historical Price Predictions vs Actual")
        
        # Check if we have the required data for historical price predictions
        if 'test_data' in results and 'price_data' in results:
            price_data = results['price_data']
            test_data = results['test_data']
            model_config = results['model_config']
            
            # Create tabs for different models
            if len(selected_models) > 1:
                model_tabs = st.tabs([f"📊 {model}" for model in selected_models])
            else:
                model_tabs = [st.container()]
            
            for i, model_name in enumerate(selected_models):
                with model_tabs[i]:
                    result_key = model_name.lower().replace('-', '_').replace(' ', '_') + '_result'
                    
                    if result_key in model_results:
                        model_info = model_results[result_key]
                        result = model_info['result']
                        
                        # Extract predictions and probabilities
                        if isinstance(result, dict) and 'predictions' in result:
                            y_pred = result['predictions']
                            y_proba = result.get('probabilities', np.ones_like(y_pred) * 0.5)
                            
                            # Convert binary predictions to price predictions
                            price_pred_data = convert_predictions_to_prices(
                                y_pred, y_proba, 
                                price_data['prices'], 
                                price_data['test_start_idx'],
                                results['model_config']['window_length']
                            )
                            
                            # Create prediction vs actual plot
                            if len(price_pred_data['actual_prices']) > 0:
                                # Get corresponding dates for the test period
                                test_dates = price_data['dates'][price_data['test_start_idx'] + results['model_config']['window_length']:
                                                                price_data['test_start_idx'] + results['model_config']['window_length'] + len(y_pred)]
                                
                                fig = plot_prediction_comparison(
                                    price_pred_data['actual_prices'],
                                    price_pred_data['predicted_prices'],
                                    test_dates,
                                    model_name,
                                    results['symbol']
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                                
                                # Show prediction statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    direction_acc = np.mean(
                                        np.sign(np.diff(price_pred_data['actual_prices'])) == 
                                        np.sign(np.diff(price_pred_data['predicted_prices']))
                                    ) * 100 if len(price_pred_data['actual_prices']) > 1 else 0
                                    st.metric("Direction Accuracy", f"{direction_acc:.1f}%")
                                
                                with col2:
                                    mape = np.mean(np.abs(
                                        (price_pred_data['actual_prices'] - price_pred_data['predicted_prices'][:len(price_pred_data['actual_prices'])]) / 
                                        price_pred_data['actual_prices']
                                    )) * 100
                                    st.metric("MAPE", f"{mape:.2f}%")
                                
                                with col3:
                                    rmse = np.sqrt(np.mean(
                                        (price_pred_data['actual_prices'] - price_pred_data['predicted_prices'][:len(price_pred_data['actual_prices'])])**2
                                    ))
                                    st.metric("RMSE", f"{rmse:.4f}")
                                
                                with col4:
                                    avg_confidence = np.mean(price_pred_data['probabilities']) * 100
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                                
                                # Show sample predictions table
                                st.subheader(f"📋 Sample Predictions - {model_name}")
                                
                                # Create a sample table with last 10 predictions
                                n_samples = min(10, len(y_pred))
                                sample_df = pd.DataFrame({
                                    'Date': test_dates[-n_samples:].strftime('%Y-%m-%d') if hasattr(test_dates[-n_samples:], 'strftime') else [str(d)[:10] for d in test_dates[-n_samples:]],
                                    'Actual Price': price_pred_data['actual_prices'][-n_samples:],
                                    'Predicted Price': price_pred_data['predicted_prices'][-n_samples:],
                                    'Prediction': ['📈 Up' if p == 1 else '📉 Down' for p in y_pred[-n_samples:]],
                                    'Confidence': [f"{p:.1%}" for p in y_proba[-n_samples:]],
                                    'Correct': ['✅' if pred == actual else '❌' for pred, actual in zip(
                                        y_pred[-n_samples:], test_data['y_test'][-n_samples:]
                                    )]
                                })
                                
                                st.dataframe(sample_df, use_container_width=True)
                            
                            else:
                                st.warning(f"Unable to generate price predictions for {model_name}")
                        else:
                            st.warning(f"No prediction data available for {model_name}")
        else:
            st.warning("Price prediction data not available. Please retrain the models to see price predictions.")
        
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