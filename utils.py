"""
Utility functions for metrics, plotting, and data handling
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from config import PLOT_DPI, PLOT_COLORS, PLOT_FIGSIZE_MEDIUM, PLOT_FIGSIZE_LARGE, OUTPUT_DIR
import os

sns.set_theme(style='whitegrid', palette='Set2')


def evaluate_model(y_true, y_pred, y_proba=None, model_name='Model'):
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        model_name: Name of the model for reporting
    
    Returns:
        dict: Dictionary with all metrics
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'model': model_name
    }
    
    return metrics


def print_metrics(metrics, y_true, y_pred):
    """Print formatted metrics and classification report."""
    print(f"\n{'='*60}")
    print(f"Model: {metrics['model']}")
    print(f"{'='*60}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Readmitted', 'Readmitted']))


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Not Readmitted', 'Readmitted']
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=1.5, linecolor='white', ax=ax,
                annot_kws={'size': 14, 'fontweight': 'bold'})
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        print(f"Saved: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name='Model', baseline_proba=None, save_path=None):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 5))
    auc = roc_auc_score(y_true, y_proba)
    
    RocCurveDisplay.from_predictions(
        y_true, y_proba, ax=ax,
        name=f'{model_name} (AUC = {auc:.3f})',
        color='steelblue', lw=2
    )
    
    if baseline_proba is not None:
        auc_base = roc_auc_score(y_true, baseline_proba)
        RocCurveDisplay.from_predictions(
            y_true, baseline_proba, ax=ax,
            name=f'Baseline (AUC = {auc_base:.3f})',
            color='orange', lw=1.5, linestyle='--'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    ax.set_title(f'ROC Curve — {model_name}', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        print(f"Saved: {save_path}")
    plt.close()


def plot_feature_importance(importances_dict, model_name='Model', top_n=20, save_path=None):
    """Plot feature importance."""
    importances = pd.Series(importances_dict)
    top = importances.sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    colors_fi = ['#d73027' if i < 5 else '#fee08b' if i < 10 else '#1a9850'
                 for i in range(len(top))]
    ax.barh(top.index[::-1], top.values[::-1], color=colors_fi[::-1], edgecolor='k')
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}', fontsize=13)
    ax.set_xlabel('Importance Score')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(results_df, save_path=None):
    """Plot metrics comparison across models."""
    fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE_LARGE)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Handle both 'model' and 'Model' column names
    model_col = 'Model' if 'Model' in results_df.columns else 'model'
    
    for ax, metric in zip(axes.flat, metrics_to_plot):
        data = results_df.sort_values(metric, ascending=True)
        ax.barh(data[model_col], data[metric], color='steelblue', edgecolor='k')
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xlim([0, 1])
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=PLOT_DPI)
        print(f"Saved: {save_path}")
    plt.close()


def find_optimal_threshold(y_true, y_proba, metric='f1', step=0.01):
    """
    Find optimal classification threshold to maximize a metric.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall', 'roc_auc')
        step: Threshold step size
    
    Returns:
        tuple: (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.1, 0.9, step)
    best_val = -1
    best_thr = 0.5
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        if metric == 'accuracy':
            val = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            val = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'roc_auc':
            val = roc_auc_score(y_true, y_proba)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if val > best_val:
            best_val = val
            best_thr = thr
    
    return best_thr, best_val


def create_results_table(all_results):
    """Create comparison table from results."""
    results_list = []
    for model_name, metrics in all_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        results_list.append(row)
    
    df = pd.DataFrame(results_list)
    return df.sort_values('recall', ascending=False)


def save_results_to_csv(results_df, filename='model_results.csv'):
    """Save results to CSV."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    return filepath


def print_header(text, char='=', width=70):
    """Print formatted header."""
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(text, char='-', width=70):
    """Print formatted section."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")
