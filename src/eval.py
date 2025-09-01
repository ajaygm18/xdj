"""
Evaluation module for computing comprehensive metrics.
Implements all evaluation metrics specified in the paper.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, matthews_corrcoef, classification_report
)
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation with all metrics from the paper."""
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "results"):
        """
        Initialize evaluator.
        
        Args:
            save_plots: Whether to save plots
            plot_dir: Directory to save plots
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        if save_plots:
            import os
            os.makedirs(plot_dir, exist_ok=True)
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray = None, model_name: str = "Model") -> Dict[str, float]:
        """
        Compute all evaluation metrics specified in the paper.
        
        Metrics computed:
        - Accuracy
        - Precision  
        - Recall
        - F1-score
        - AUC-ROC
        - PR-AUC (Precision-Recall AUC)
        - MCC (Matthews Correlation Coefficient)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for AUC metrics)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # AUC metrics (require probabilities)
        if y_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            except ValueError as e:
                print(f"Warning: Could not compute AUC metrics for {model_name}: {e}")
                metrics['auc_roc'] = np.nan
                metrics['pr_auc'] = np.nan
        else:
            metrics['auc_roc'] = np.nan
            metrics['pr_auc'] = np.nan
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], model_name: str = "Model") -> None:
        """Print formatted metrics."""
        print(f"\n=== {model_name} Evaluation Metrics ===")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1-Score:    {metrics['f1_score']:.4f}")
        print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
        print(f"MCC:         {metrics['mcc']:.4f}")
        print("=" * 40)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Model name for title
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'{model_name} - Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = f'{model_name} - Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                   xticklabels=['Downward', 'Upward'],
                   yticklabels=['Downward', 'Upward'])
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        if self.save_plots:
            filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            fig.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str = "Model") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Model name for title
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} - ROC Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            filename = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
            fig.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   model_name: str = "Model") -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Model name for title
            
        Returns:
            Matplotlib figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        ax.axhline(y=np.mean(y_true), color='k', linestyle='--', linewidth=1, 
                  label=f'Random Classifier (PR-AUC = {np.mean(y_true):.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{model_name} - Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            filename = f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
            fig.savefig(f"{self.plot_dir}/{filename}", dpi=300, bbox_inches='tight')
        
        return fig
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = "Model", is_torch_model: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Model name for reporting
            is_torch_model: Whether the model is a PyTorch model
            
        Returns:
            Dictionary containing metrics and plots
        """
        print(f"\nEvaluating {model_name}...")
        
        # Get predictions
        if is_torch_model:
            y_pred, y_proba = self._predict_torch_model(model, X_test)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Compute metrics
        metrics = self.compute_all_metrics(y_test, y_pred, y_proba, model_name)
        self.print_metrics(metrics, model_name)
        
        # Create plots
        plots = {}
        plots['confusion_matrix'] = self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        if y_proba is not None:
            plots['roc_curve'] = self.plot_roc_curve(y_test, y_proba, model_name)
            plots['pr_curve'] = self.plot_precision_recall_curve(y_test, y_proba, model_name)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'plots': plots
        }
    
    def _predict_torch_model(self, model: nn.Module, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with PyTorch model."""
        model.eval()
        device = next(model.parameters()).device
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).to(device)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 128
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                
                # Forward pass
                model_output = model(batch)
                if isinstance(model_output, tuple):
                    # PLSTM-TAL returns (logits, attention_weights)
                    logits, _ = model_output
                else:
                    # Baseline models return only logits
                    logits = model_output
                
                # Convert to probabilities and predictions
                proba = torch.sigmoid(logits).cpu().numpy()
                pred = (proba > 0.5).astype(int)
                
                predictions.extend(pred)
                probabilities.extend(proba)
        
        return np.array(predictions), np.array(probabilities)
    
    def compare_models(self, results: Dict[str, Dict], metric: str = 'accuracy') -> pd.DataFrame:
        """
        Compare multiple models on a specific metric.
        
        Args:
            results: Dictionary of evaluation results for each model
            metric: Metric to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc_roc'],
                'PR-AUC': metrics['pr_auc'],
                'MCC': metrics['mcc']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values(by=metric.title(), ascending=False)
        
        return df
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                             metrics: List[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models across metrics.
        
        Args:
            results: Dictionary of evaluation results for each model
            metrics: List of metrics to plot
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'pr_auc', 'mcc']
        
        # Prepare data
        models = list(results.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for model_name in models:
            model_metrics = results[model_name]['metrics']
            for metric in metrics:
                metric_values[metric].append(model_metrics.get(metric, 0))
        
        # Create plot
        x = np.arange(len(models))
        width = 0.1
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics) / 2) * width
            ax.bar(x + offset, metric_values[metric], width, label=metric.upper().replace('_', '-'))
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if self.save_plots:
            fig.savefig(f"{self.plot_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test evaluation module
    print("Testing evaluation module...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic predictions and labels
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.beta(2, 2, n_samples)  # Random probabilities
    y_pred = (y_proba > 0.5).astype(int)
    
    # Add some correlation to make results more realistic
    for i in range(len(y_true)):
        if y_true[i] == 1:
            y_proba[i] = min(1.0, y_proba[i] + 0.3)  # Increase probability for positive cases
        else:
            y_proba[i] = max(0.0, y_proba[i] - 0.3)  # Decrease probability for negative cases
    
    y_pred = (y_proba > 0.5).astype(int)
    
    print(f"Test data: {len(y_true)} samples")
    print(f"True label distribution: {np.bincount(y_true)}")
    print(f"Predicted label distribution: {np.bincount(y_pred)}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(save_plots=False)  # Don't save plots in test
    
    # Compute metrics
    metrics = evaluator.compute_all_metrics(y_true, y_pred, y_proba, "Test Model")
    evaluator.print_metrics(metrics, "Test Model")
    
    # Test plotting functions
    print("\nTesting plotting functions...")
    
    fig1 = evaluator.plot_confusion_matrix(y_true, y_pred, "Test Model")
    print("Confusion matrix plot created")
    
    fig2 = evaluator.plot_roc_curve(y_true, y_proba, "Test Model")
    print("ROC curve plot created")
    
    fig3 = evaluator.plot_precision_recall_curve(y_true, y_proba, "Test Model")
    print("PR curve plot created")
    
    # Close plots to free memory
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    print("\nEvaluation module testing completed successfully!")