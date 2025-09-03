"""
Bayesian Hyperparameter Optimization for PLSTM-TAL model.
Implements paper-compliant hyperparameter optimization using Gaussian Process optimization.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# Try different import methods for the modules
try:
    from .model_plstm_tal import PLSTM_TAL
    from .train import ModelTrainer
except ImportError:
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from model_plstm_tal import PLSTM_TAL
        from train import ModelTrainer
    except ImportError:
        # For standalone testing - will be handled in __main__
        PLSTM_TAL = None
        ModelTrainer = None


class BayesianOptimizer:
    """Bayesian hyperparameter optimization for PLSTM-TAL model using Gaussian Process."""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_size, n_calls=50, random_state=42):
        """
        Initialize Bayesian optimizer.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            input_size: Input feature size
            n_calls: Number of optimization calls (paper suggests 50-100)
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = input_size
        self.n_calls = n_calls
        self.random_state = random_state
        
        # Define search space based on paper specifications and reasonable ranges
        self.search_space = [
            # PLSTM-TAL specific parameters
            Integer(32, 128, name='hidden_size'),  # Paper uses 64, but allow optimization around it
            Real(0.05, 0.3, name='dropout'),      # Paper uses 0.1, but allow optimization
            Integer(1, 3, name='num_layers'),     # Paper uses 1, but allow optimization
            
            # Training parameters
            Real(1e-4, 1e-2, name='learning_rate'), # Optimize learning rate
            Integer(16, 64, name='batch_size'),     # Optimize batch size
            Integer(10, 40, name='window_length'),   # Optimize sequence length
            
            # CAE parameters (if needed)
            Integer(32, 128, name='cae_hidden_dim'),
            Integer(8, 32, name='cae_encoding_dim'),
            Real(1e-5, 1e-3, name='cae_lambda_reg'),
            
            # Optimizer choice (paper prefers Adamax)
            Categorical(['adam', 'adamax', 'rmsprop'], name='optimizer'),
            
            # Activation function (paper uses tanh)
            Categorical(['tanh', 'relu', 'sigmoid'], name='activation')
        ]
        
        self.best_params = None
        self.best_score = float('-inf')
        self.optimization_history = []
    
    def objective_function(self, **params):
        """
        Objective function for Bayesian optimization.
        Returns negative validation accuracy (since we minimize).
        """
        try:
            # Create model with current hyperparameters
            model = PLSTM_TAL(
                input_size=self.input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                activation=params['activation']
            )
            
            # Train model with current hyperparameters
            trainer = ModelTrainer(model)
            history = trainer.train(
                self.X_train, self.y_train, 
                self.X_val, self.y_val,
                epochs=50,  # Reduced for optimization speed
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer_name=params['optimizer'],
                early_stopping_patience=10,
                verbose=False  # Reduce output during optimization
            )
            
            # Get best validation accuracy
            val_accuracy = max(history['val_accuracy'])
            
            # Store current result
            result = {
                'params': params.copy(),
                'val_accuracy': val_accuracy,
                'loss': -val_accuracy  # Negative for minimization
            }
            self.optimization_history.append(result)
            
            # Update best if current is better
            if val_accuracy > self.best_score:
                self.best_score = val_accuracy
                self.best_params = params.copy()
                print(f"New best validation accuracy: {val_accuracy:.4f}")
                print(f"Best params: {params}")
            
            return -val_accuracy  # Return negative for minimization
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1.0  # Return large positive value for failed runs
    
    def optimize(self, verbose=True):
        """
        Run Bayesian optimization to find best hyperparameters.
        
        Returns:
            dict: Best hyperparameters found
        """
        if verbose:
            print("Starting Bayesian hyperparameter optimization...")
            print(f"Search space: {len(self.search_space)} dimensions")
            print(f"Number of evaluations: {self.n_calls}")
            print(f"Training samples: {len(self.X_train)}")
            print(f"Validation samples: {len(self.X_val)}")
        
        # Set the search space for the objective function and create decorated version
        decorated_objective = use_named_args(self.search_space)(self.objective_function)
        
        # Run Gaussian Process optimization  
        try:
            result = gp_minimize(
                func=decorated_objective,
                dimensions=self.search_space,
                n_calls=max(self.n_calls, 10),  # Ensure minimum 10 calls
                random_state=self.random_state,
                acq_func='EI',  # Expected Improvement
                n_initial_points=min(5, max(2, self.n_calls // 2)),  # Adaptive initial points
                verbose=verbose
            )
        except Exception as e:
            if "n_calls" in str(e) and self.n_calls < 10:
                print(f"⚠️ Adjusting n_calls from {self.n_calls} to 10 (minimum requirement)")
                result = gp_minimize(
                    func=decorated_objective,
                    dimensions=self.search_space,
                    n_calls=10,
                    random_state=self.random_state,
                    acq_func='EI',
                    n_initial_points=5,
                    verbose=verbose
                )
            else:
                raise e
        
        # Extract best parameters
        best_params_list = result.x
        best_params = {}
        for i, param_name in enumerate([dim.name for dim in self.search_space]):
            best_params[param_name] = best_params_list[i]
        
        self.best_params = best_params
        self.best_score = -result.fun  # Convert back to positive
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best validation accuracy: {self.best_score:.4f}")
            print(f"Best hyperparameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        return self.best_params
    
    def get_paper_compliant_params(self):
        """
        Get paper-compliant hyperparameters as baseline.
        These are the exact parameters mentioned in the research paper.
        """
        return {
            'hidden_size': 64,        # Paper specification
            'dropout': 0.1,           # Paper specification  
            'num_layers': 1,          # Paper specification
            'learning_rate': 1e-3,    # Standard value
            'batch_size': 32,         # Standard value
            'window_length': 20,      # Standard value
            'cae_hidden_dim': 64,     # Reasonable value
            'cae_encoding_dim': 16,   # Compression ratio
            'cae_lambda_reg': 1e-4,   # Standard regularization
            'optimizer': 'adamax',    # Paper specification
            'activation': 'tanh'      # Paper specification
        }
    
    def compare_with_paper_params(self):
        """
        Compare optimized parameters with paper-compliant parameters.
        """
        if self.best_params is None:
            print("No optimization results available. Run optimize() first.")
            return
        
        paper_params = self.get_paper_compliant_params()
        
        print("\n" + "="*60)
        print("HYPERPARAMETER COMPARISON: OPTIMIZED vs PAPER")
        print("="*60)
        print(f"{'Parameter':<20} {'Paper':<15} {'Optimized':<15} {'Change':<15}")
        print("-"*60)
        
        for param in paper_params.keys():
            if param in self.best_params:
                paper_val = paper_params[param]
                opt_val = self.best_params[param]
                
                if isinstance(paper_val, (int, float)) and isinstance(opt_val, (int, float)):
                    if paper_val != 0:
                        change = f"{((opt_val - paper_val) / paper_val * 100):+.1f}%"
                    else:
                        change = "N/A"
                else:
                    change = "Changed" if paper_val != opt_val else "Same"
                
                print(f"{param:<20} {str(paper_val):<15} {str(opt_val):<15} {change:<15}")
        
        print("-"*60)
        print(f"Best validation accuracy: {self.best_score:.4f}")
        print("="*60)
    
    def get_optimization_summary(self):
        """Get summary of optimization process."""
        if not self.optimization_history:
            return "No optimization results available."
        
        accuracies = [result['val_accuracy'] for result in self.optimization_history]
        
        summary = {
            'n_evaluations': len(self.optimization_history),
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'improvement': max(accuracies) - accuracies[0] if len(accuracies) > 1 else 0
        }
        
        return summary


class QuickBayesianOptimizer(BayesianOptimizer):
    """Faster version of Bayesian optimizer for testing purposes."""
    
    def __init__(self, X_train, y_train, X_val, y_val, input_size, n_calls=10, random_state=42):
        """Initialize with fewer calls for faster optimization."""
        super().__init__(X_train, y_train, X_val, y_val, input_size, n_calls, random_state)
        
        # Reduced search space for faster optimization
        self.search_space = [
            Integer(32, 96, name='hidden_size'),   # Narrower range around paper value
            Real(0.05, 0.2, name='dropout'),       # Narrower range around paper value
            Integer(1, 2, name='num_layers'),      # Fewer options
            Real(5e-4, 5e-3, name='learning_rate'), # Narrower range
            Integer(16, 32, name='batch_size'),     # Fewer options
            Categorical(['adamax', 'adam'], name='optimizer'), # Paper preference
            Categorical(['tanh'], name='activation')  # Paper specification only
        ]
    
    def objective_function(self, **params):
        """Faster objective function with reduced training epochs."""
        try:
            # Add missing params with paper defaults
            full_params = {
                'window_length': 20,
                'cae_hidden_dim': 64,
                'cae_encoding_dim': 16,
                'cae_lambda_reg': 1e-4,
                'activation': 'tanh'
            }
            full_params.update(params)
            
            # Create model
            model = PLSTM_TAL(
                input_size=self.input_size,
                hidden_size=full_params['hidden_size'],
                num_layers=full_params['num_layers'],
                dropout=full_params['dropout'],
                activation=full_params['activation']
            )
            
            # Train with fewer epochs for speed
            trainer = ModelTrainer(model)
            history = trainer.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=20,  # Much fewer epochs
                batch_size=full_params['batch_size'],
                learning_rate=full_params['learning_rate'],
                optimizer_name=full_params['optimizer'],
                early_stopping_patience=5,
                verbose=False
            )
            
            val_accuracy = max(history['val_accuracy'])
            
            # Store result
            result = {
                'params': full_params.copy(),
                'val_accuracy': val_accuracy,
                'loss': -val_accuracy
            }
            self.optimization_history.append(result)
            
            if val_accuracy > self.best_score:
                self.best_score = val_accuracy
                self.best_params = full_params.copy()
                print(f"✓ New best: {val_accuracy:.4f} with {full_params}")
            
            return -val_accuracy
            
        except Exception as e:
            print(f"Error: {e}")
            return 1.0


if __name__ == "__main__":
    # Test Bayesian optimization with synthetic data
    print("Testing Bayesian Hyperparameter Optimization...")
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples, seq_len, n_features = 1000, 20, 16
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    
    print(f"Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Test quick optimizer
    print("\nTesting Quick Bayesian Optimizer...")
    optimizer = QuickBayesianOptimizer(
        X_train, y_train, X_val, y_val, 
        input_size=n_features, 
        n_calls=5  # Very few for testing
    )
    
    best_params = optimizer.optimize(verbose=True)
    optimizer.compare_with_paper_params()
    
    print(f"\nOptimization summary: {optimizer.get_optimization_summary()}")
    print("✅ Bayesian optimization test completed!")