"""
Visualization functions for regression models.

This module provides plotting utilities for regression fits, residuals,
kernel comparisons, and error metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


def plot_regression_fit(X, y, y_pred, title="Regression Fit", xlabel="X", ylabel="y"):
    """
    Plot data points and regression fit.
    
    Parameters:
    -----------
    X : array (n_samples,) or (n_samples, 1)
        Feature values
    y : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    if X.ndim > 1:
        X = X.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort for plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    # Plot data and fit
    ax.scatter(X, y, alpha=0.6, s=50, label='Data', color='blue', edgecolors='black')
    ax.plot(X_sorted, y_pred_sorted, 'r-', linewidth=2, label='Fit')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_residuals(y_true, y_pred, figsize=(14, 5)):
    """
    Plot residual analysis (residuals vs predicted, histogram).
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .error_metrics import residuals, standardized_residuals
    
    res = residuals(y_true, y_pred)
    std_res = standardized_residuals(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, res, alpha=0.6, edgecolors='black')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11)
    axes[0].set_ylabel('Residuals', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(res, bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot (standardized residuals)
    from scipy import stats
    stats.probplot(std_res, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_kernel_comparison(X, y, models, X_test=None, figsize=(14, 5)):
    """
    Compare different kernel regression models.
    
    Parameters:
    -----------
    X : array (n_samples,)
        Training features
    y : array (n_samples,)
        Training targets
    models : dict
        Dictionary of {name: model} pairs
    X_test : array, optional
        Test points for prediction
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from .kernel_regression import kernel_regression_predict
    
    if X_test is None:
        X_test = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.scatter(X, y, alpha=0.6, s=50, label='Data', color='blue', 
              edgecolors='black', zorder=3)
    
    # Plot each model
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (name, model) in enumerate(models.items()):
        y_pred = kernel_regression_predict(X_test, model)
        ax.plot(X_test, y_pred, linewidth=2, label=name, 
               color=colors[i % len(colors)])
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Kernel Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_polynomial_degrees(X, y, degrees, lambda_reg=0.0, figsize=(14, 5)):
    """
    Compare polynomial regression for different degrees.
    
    Parameters:
    -----------
    X : array (n_samples,)
        Features
    y : array (n_samples,)
        Targets
    degrees : list
        List of degrees to compare
    lambda_reg : float
        Regularization parameter
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from .polynomial_regression import polynomial_fit, polynomial_predict
    
    X_test = np.linspace(X.min() - 0.5, X.max() + 0.5, 200)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    ax.scatter(X, y, alpha=0.6, s=50, label='Data', color='blue',
              edgecolors='black', zorder=3)
    
    # Plot each degree
    colors = cm.viridis(np.linspace(0, 1, len(degrees)))
    for degree, color in zip(degrees, colors):
        model = polynomial_fit(X, y, degree=degree, lambda_reg=lambda_reg)
        y_pred = polynomial_predict(X_test, model)
        ax.plot(X_test, y_pred, linewidth=2, label=f'Degree {degree}', color=color)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Polynomial Degree Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_regularization_path(X, y, lambdas, reg_type='ridge', figsize=(14, 5)):
    """
    Plot regularization path (coefficient values vs lambda).
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Features
    y : array (n_samples,)
        Targets
    lambdas : array
        Regularization parameters to try
    reg_type : str
        'ridge' or 'lasso'
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .linear_regression import ridge_fit, lasso_fit
    
    # Fit models for each lambda
    weights_history = []
    for lam in lambdas:
        if reg_type == 'ridge':
            w = ridge_fit(X, y, lambda_reg=lam)
        else:
            w = lasso_fit(X, y, lambda_reg=lam)
        weights_history.append(w)
    
    weights_history = np.array(weights_history)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot coefficient paths
    for i in range(1, weights_history.shape[1]):  # Skip intercept
        axes[0].plot(lambdas, weights_history[:, i], linewidth=2, 
                    label=f'w_{i}')
    
    axes[0].set_xlabel('λ (Regularization)', fontsize=11)
    axes[0].set_ylabel('Coefficient Value', fontsize=11)
    axes[0].set_title(f'{reg_type.capitalize()} Regularization Path', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot L1/L2 norm of weights
    l1_norms = np.sum(np.abs(weights_history[:, 1:]), axis=1)
    l2_norms = np.sqrt(np.sum(weights_history[:, 1:]**2, axis=1))
    
    axes[1].plot(lambdas, l1_norms, 'b-', linewidth=2, label='L1 Norm')
    axes[1].plot(lambdas, l2_norms, 'r-', linewidth=2, label='L2 Norm')
    axes[1].set_xlabel('λ (Regularization)', fontsize=11)
    axes[1].set_ylabel('Norm of Weights', fontsize=11)
    axes[1].set_title('Weight Norms vs λ', fontsize=12, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_error_metrics(y_true, y_pred, figsize=(12, 8)):
    """
    Visualize multiple error metrics.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .error_metrics import compute_all_metrics
    
    metrics = compute_all_metrics(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # True vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='black')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', 
                   linewidth=2, label='Perfect Prediction')
    
    axes[0, 0].set_xlabel('True Values', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Values', fontsize=11)
    axes[0, 0].set_title('True vs Predicted', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Metrics bar chart
    metric_names = ['MAE', 'RMSE', 'NSE', 'R²']
    metric_values = [metrics[name] for name in metric_names]
    
    axes[0, 1].barh(metric_names, metric_values, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Value', fontsize=11)
    axes[0, 1].set_title('Error Metrics', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Residuals
    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Predicted Values', fontsize=11)
    axes[1, 0].set_ylabel('Residuals', fontsize=11)
    axes[1, 0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics text
    axes[1, 1].axis('off')
    metrics_text = "Error Metrics:\n\n"
    for name, value in metrics.items():
        if name not in ['Max Error', 'Median AE']:
            metrics_text += f"{name:12s}: {value:8.4f}\n"
    
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    return fig, axes


def plot_learning_curve(train_sizes, train_scores, val_scores, figsize=(10, 6)):
    """
    Plot learning curve (performance vs training set size).
    
    Parameters:
    -----------
    train_sizes : array
        Training set sizes
    train_scores : array
        Training scores
    val_scores : array
        Validation scores
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(train_sizes, train_scores, 'o-', linewidth=2, label='Training', color='blue')
    ax.plot(train_sizes, val_scores, 'o-', linewidth=2, label='Validation', color='red')
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Score (R²)', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_bias_variance_tradeoff(degrees, train_errors, val_errors, figsize=(10, 6)):
    """
    Plot bias-variance tradeoff.
    
    Parameters:
    -----------
    degrees : array
        Model complexities (e.g., polynomial degrees)
    train_errors : array
        Training errors
    val_errors : array
        Validation errors
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(degrees, train_errors, 'o-', linewidth=2, label='Training Error', color='blue')
    ax.plot(degrees, val_errors, 'o-', linewidth=2, label='Validation Error', color='red')
    
    # Mark optimal point
    optimal_idx = np.argmin(val_errors)
    ax.plot(degrees[optimal_idx], val_errors[optimal_idx], 'g*', 
           markersize=15, label='Optimal')
    
    ax.set_xlabel('Model Complexity (Degree)', fontsize=12)
    ax.set_ylabel('Error (RMSE)', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax
