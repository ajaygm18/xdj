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
    TALIB_AVAILABLE = True
except ImportError:
    from src.indicators import TechnicalIndicators
    TALIB_AVAILABLE = False

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
        ["Paper Default (2005-2022)", "Last 5 Years", "Last 3 Years", "Last 1 Year", "Custom"]
    )
    
    if date_preset == "Paper Default (2005-2022)":
        start_date = "2005-01-01"
        end_date = "2022-03-31"
    elif date_preset == "Last 5 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 3 Years":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    elif date_preset == "Last 1 Year":
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:  # Custom
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1)).strftime("%Y-%m-%d")
        with col2:
            end_date = st.date_input("End Date", value=datetime.now()).strftime("%Y-%m-%d")
    
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
        
        if st.button("🚀 Train PLSTM-TAL Model", type="primary"):
            train_model(st.session_state.stock_data, symbol, model_config, use_bayesian, use_quick_mode, save_model)
    
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
                indicators = TechnicalIndicatorsTA()
                st.info("Using TA-Lib for paper-compliant indicators")
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

def display_predictions():
    """Display model predictions and accuracy metrics."""
    
    results = st.session_state.prediction_results
    plstm_result = results['plstm_result']
    baseline_results = results['baseline_results']
    
    st.subheader("🎯 Model Performance")
    
    # Main metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "PLSTM-TAL Accuracy", 
            f"{plstm_result['metrics']['accuracy']:.3f}",
            delta=f"+{(plstm_result['metrics']['accuracy'] - baseline_results['LSTM']['metrics']['accuracy']):.3f}"
        )
    
    with col2:
        st.metric(
            "Precision", 
            f"{plstm_result['metrics']['precision']:.3f}"
        )
    
    with col3:
        st.metric(
            "Recall", 
            f"{plstm_result['metrics']['recall']:.3f}"
        )
    
    with col4:
        st.metric(
            "F1 Score", 
            f"{plstm_result['metrics']['f1_score']:.3f}"
        )
    
    # Model comparison chart
    st.subheader("📊 Model Comparison")
    
    comparison_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'PLSTM-TAL': plstm_result['metrics'][metric],
            'LSTM': baseline_results['LSTM']['metrics'][metric],
            'SVM': baseline_results['SVM']['metrics'][metric]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_data))
    width = 0.25
    
    ax.bar(x - width, comparison_df['PLSTM-TAL'], width, label='PLSTM-TAL', color='#1f77b4')
    ax.bar(x, comparison_df['LSTM'], width, label='LSTM', color='#ff7f0e')
    ax.bar(x + width, comparison_df['SVM'], width, label='SVM', color='#2ca02c')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Metric'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(comparison_df.set_index('Metric'))

def display_detailed_results():
    """Display detailed results and analysis."""
    
    results = st.session_state.prediction_results
    plstm_result = results['plstm_result']
    data_info = results['data_info']
    model_config = results['model_config']
    
    # Data information
    st.subheader("📊 Dataset Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"""
        **Stock Symbol:** {data_info['symbol']}
        **Total Samples:** {data_info['total_samples']}
        **Features:** {data_info['features_count']}
        """)
    
    with col2:
        st.info(f"""
        **Training Samples:** {data_info['training_samples']}
        **Test Samples:** {data_info['test_samples']}
        **Test Split:** {data_info['test_samples'] / (data_info['training_samples'] + data_info['test_samples']):.2%}
        """)
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    config_df = pd.DataFrame(list(model_config.items()), columns=['Parameter', 'Value'])
    st.dataframe(config_df)
    
    # All metrics
    st.subheader("📈 All Performance Metrics")
    
    metrics_data = []
    for name, result in [('PLSTM-TAL', plstm_result)] + list(results['baseline_results'].items()):
        row = {'Model': name}
        row.update(result['metrics'])
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index('Model'))
    
    # Download results
    st.subheader("💾 Download Results")
    
    if st.button("📥 Download Results as JSON"):
        results_json = {
            'model_performance': {name: result['metrics'] for name, result in [('PLSTM-TAL', plstm_result)] + list(results['baseline_results'].items())},
            'model_config': model_config,
            'data_info': data_info,
            'timestamp': datetime.now().isoformat()
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(results_json, indent=2),
            file_name=f"plstm_tal_results_{data_info['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()