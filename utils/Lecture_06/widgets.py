"""
Interactive widgets for regression exploration.

This module provides ipywidgets for exploring linear regression,
polynomial degrees, kernels, regularization, and error metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Widgets will not work.")


def linear_regression_widget():
    """
    Interactive widget for exploring linear regression.
    
    Allows adjustment of:
    - Noise level
    - Number of samples
    - Regularization (Ridge/LASSO)
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return
    
    from .linear_regression import ridge_fit, linear_predict
    from .visualizations import plot_regression_fit
    from .error_metrics import compute_all_metrics
    
    def update(n_samples=50, noise=0.5, lambda_reg=0.0, reg_type='None'):
        # Generate data
        np.random.seed(42)
        X = np.random.uniform(-3, 3, n_samples)
        y_true = 2 * X + 1
        y = y_true + np.random.normal(0, noise, n_samples)
        
        # Fit model
        X_2d = X.reshape(-1, 1)
        if reg_type == 'None':
            w = ridge_fit(X_2d, y, lambda_reg=0.0)
        else:
            w = ridge_fit(X_2d, y, lambda_reg=lambda_reg)
        
        # Predict
        y_pred = linear_predict(X_2d, w)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, s=50, label='Data', edgecolors='black')
        
        X_line = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_line = linear_predict(X_line, w)
        plt.plot(X_line, y_line, 'r-', linewidth=2, label='Fit')
        plt.plot(X, y_true, 'g--', linewidth=2, alpha=0.5, label='True')
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'Linear Regression ({reg_type})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        metrics = compute_all_metrics(y, y_pred)
        metrics_text = f"Model: y = {w[1]:.3f}x + {w[0]:.3f}\n\n"
        metrics_text += "Metrics:\n"
        for name, value in metrics.items():
            if name not in ['Max Error', 'Median AE']:
                metrics_text += f"{name:10s}: {value:8.4f}\n"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create widgets
    n_samples_slider = widgets.IntSlider(value=50, min=10, max=200, step=10,
                                         description='Samples:')
    noise_slider = widgets.FloatSlider(value=0.5, min=0.0, max=2.0, step=0.1,
                                       description='Noise:')
    lambda_slider = widgets.FloatSlider(value=0.0, min=0.0, max=10.0, step=0.5,
                                        description='λ:')
    reg_dropdown = widgets.Dropdown(options=['None', 'Ridge'],
                                    value='None', description='Reg Type:')
    
    widgets.interact(update, n_samples=n_samples_slider, noise=noise_slider,
                    lambda_reg=lambda_slider, reg_type=reg_dropdown)


def polynomial_degree_widget():
    """
    Interactive widget for exploring polynomial regression degrees.
    
    Allows adjustment of:
    - Polynomial degree
    - Regularization
    - Noise level
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return
    
    from .polynomial_regression import polynomial_fit, polynomial_predict
    from .error_metrics import compute_all_metrics
    
    def update(degree=1, lambda_reg=0.0, noise=0.3):
        # Generate data
        np.random.seed(42)
        X = np.linspace(-3, 3, 30)
        y_true = 0.5 * X**2 - X + 1
        y = y_true + np.random.normal(0, noise, len(X))
        
        # Fit model
        model = polynomial_fit(X, y, degree=degree, lambda_reg=lambda_reg)
        
        # Predict
        X_test = np.linspace(-3, 3, 200)
        y_pred_train = polynomial_predict(X, model)
        y_pred_test = polynomial_predict(X_test, model)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, s=50, label='Data', edgecolors='black')
        plt.plot(X_test, y_pred_test, 'r-', linewidth=2, label=f'Degree {degree}')
        plt.plot(X, y_true, 'g--', linewidth=2, alpha=0.5, label='True')
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'Polynomial Regression (Degree {degree})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 10)
        
        # Metrics
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        metrics = compute_all_metrics(y, y_pred_train)
        metrics_text = f"Degree: {degree}\nλ: {lambda_reg}\n\n"
        metrics_text += "Training Metrics:\n"
        for name, value in metrics.items():
            if name not in ['Max Error', 'Median AE']:
                metrics_text += f"{name:10s}: {value:8.4f}\n"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create widgets
    degree_slider = widgets.IntSlider(value=1, min=1, max=10, step=1,
                                     description='Degree:')
    lambda_slider = widgets.FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1,
                                        description='λ:')
    noise_slider = widgets.FloatSlider(value=0.3, min=0.0, max=2.0, step=0.1,
                                       description='Noise:')
    
    widgets.interact(update, degree=degree_slider, lambda_reg=lambda_slider,
                    noise=noise_slider)


def kernel_explorer_widget():
    """
    Interactive widget for exploring kernel regression.
    
    Allows adjustment of:
    - Kernel type (linear, polynomial, RBF)
    - Kernel parameters
    - Regularization
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return
    
    from .kernel_regression import kernel_regression_fit, kernel_regression_predict
    from .error_metrics import compute_all_metrics
    
    def update(kernel_type='rbf', sigma=1.0, degree=2, lambda_reg=1.0):
        # Generate data
        np.random.seed(42)
        X = np.linspace(-3, 3, 30)
        y = np.sin(X) + 0.3 * np.random.randn(len(X))
        
        # Fit model
        if kernel_type == 'rbf':
            model = kernel_regression_fit(X, y, kernel_type='rbf', 
                                         lambda_reg=lambda_reg, sigma=sigma)
        elif kernel_type == 'polynomial':
            model = kernel_regression_fit(X, y, kernel_type='polynomial',
                                         lambda_reg=lambda_reg, degree=int(degree))
        else:
            model = kernel_regression_fit(X, y, kernel_type='linear',
                                         lambda_reg=lambda_reg)
        
        # Predict
        X_test = np.linspace(-3, 3, 200)
        y_pred_train = kernel_regression_predict(X, model)
        y_pred_test = kernel_regression_predict(X_test, model)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, s=50, label='Data', edgecolors='black', zorder=3)
        plt.plot(X_test, y_pred_test, 'r-', linewidth=2, label=f'{kernel_type.upper()} Kernel')
        plt.plot(X_test, np.sin(X_test), 'g--', linewidth=2, alpha=0.5, label='True')
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'Kernel Regression ({kernel_type.upper()})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        metrics = compute_all_metrics(y, y_pred_train)
        metrics_text = f"Kernel: {kernel_type}\n"
        if kernel_type == 'rbf':
            metrics_text += f"σ: {sigma:.2f}\n"
        elif kernel_type == 'polynomial':
            metrics_text += f"Degree: {int(degree)}\n"
        metrics_text += f"λ: {lambda_reg:.2f}\n\n"
        metrics_text += "Metrics:\n"
        for name, value in metrics.items():
            if name not in ['Max Error', 'Median AE']:
                metrics_text += f"{name:10s}: {value:8.4f}\n"
        
        plt.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create widgets
    kernel_dropdown = widgets.Dropdown(options=['linear', 'polynomial', 'rbf'],
                                      value='rbf', description='Kernel:')
    sigma_slider = widgets.FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1,
                                       description='σ (RBF):')
    degree_slider = widgets.IntSlider(value=2, min=1, max=5, step=1,
                                     description='Degree (Poly):')
    lambda_slider = widgets.FloatSlider(value=1.0, min=0.01, max=10.0, step=0.5,
                                        description='λ:')
    
    widgets.interact(update, kernel_type=kernel_dropdown, sigma=sigma_slider,
                    degree=degree_slider, lambda_reg=lambda_slider)


def regularization_widget():
    """
    Interactive widget for comparing L1 (LASSO) and L2 (Ridge) regularization.
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return
    
    from .linear_regression import ridge_fit, lasso_fit, linear_predict
    
    def update(lambda_reg=1.0):
        # Generate data with many features
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        
        # Only first 3 features are relevant
        true_weights = np.array([2, -1, 0.5] + [0] * 7)
        y = X @ true_weights + 0.5 * np.random.randn(n_samples)
        
        # Fit models
        w_ridge = ridge_fit(X, y, lambda_reg=lambda_reg)
        w_lasso = lasso_fit(X, y, lambda_reg=lambda_reg)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        # Coefficient comparison
        plt.subplot(1, 2, 1)
        x_pos = np.arange(1, n_features + 1)
        width = 0.35
        
        plt.bar(x_pos - width/2, w_ridge[1:], width, label='Ridge', alpha=0.7)
        plt.bar(x_pos + width/2, w_lasso[1:], width, label='LASSO', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.5)
        
        plt.xlabel('Feature Index', fontsize=12)
        plt.ylabel('Coefficient Value', fontsize=12)
        plt.title(f'Regularization Comparison (λ={lambda_reg})', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Sparsity comparison
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        n_nonzero_ridge = np.sum(np.abs(w_ridge[1:]) > 0.01)
        n_nonzero_lasso = np.sum(np.abs(w_lasso[1:]) > 0.01)
        
        text = f"λ = {lambda_reg}\n\n"
        text += "Ridge (L2):\n"
        text += f"  Non-zero: {n_nonzero_ridge}/{n_features}\n"
        text += f"  L1 norm: {np.sum(np.abs(w_ridge[1:])):.3f}\n"
        text += f"  L2 norm: {np.sqrt(np.sum(w_ridge[1:]**2)):.3f}\n\n"
        text += "LASSO (L1):\n"
        text += f"  Non-zero: {n_nonzero_lasso}/{n_features}\n"
        text += f"  L1 norm: {np.sum(np.abs(w_lasso[1:])):.3f}\n"
        text += f"  L2 norm: {np.sqrt(np.sum(w_lasso[1:]**2)):.3f}\n"
        
        plt.text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create widget
    lambda_slider = widgets.FloatSlider(value=1.0, min=0.0, max=10.0, step=0.5,
                                        description='λ:')
    
    widgets.interact(update, lambda_reg=lambda_slider)


def error_metric_widget():
    """
    Interactive widget for comparing error metrics.
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available")
        return
    
    from .error_metrics import mae, rmse, nse
    
    def update(outlier_strength=0.0):
        # Generate data
        np.random.seed(42)
        X = np.linspace(0, 10, 50)
        y_true = 2 * X + 1
        y_pred = y_true + np.random.normal(0, 1, len(X))
        
        # Add outlier
        if outlier_strength > 0:
            y_pred[25] += outlier_strength
        
        # Compute metrics
        mae_val = mae(y_true, y_pred)
        rmse_val = rmse(y_true, y_pred)
        nse_val = nse(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X, y_true, alpha=0.6, s=50, label='True', edgecolors='black')
        plt.scatter(X, y_pred, alpha=0.6, s=50, label='Predicted', edgecolors='black')
        
        if outlier_strength > 0:
            plt.scatter(X[25], y_pred[25], s=200, color='red', marker='*',
                       label='Outlier', zorder=5)
        
        plt.xlabel('X', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title('True vs Predicted', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        text = f"Outlier Strength: {outlier_strength}\n\n"
        text += "Error Metrics:\n"
        text += f"MAE:  {mae_val:.4f}\n"
        text += f"RMSE: {rmse_val:.4f}\n"
        text += f"NSE:  {nse_val:.4f}\n\n"
        text += "Observations:\n"
        text += "• RMSE ≥ MAE always\n"
        text += "• RMSE more sensitive\n"
        text += "  to outliers\n"
        text += "• NSE compares to\n"
        text += "  mean baseline"
        
        plt.text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
    
    # Create widget
    outlier_slider = widgets.FloatSlider(value=0.0, min=0.0, max=20.0, step=1.0,
                                         description='Outlier:')
    
    widgets.interact(update, outlier_strength=outlier_slider)
