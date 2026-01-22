"""
Interactive widgets for exploring optimization concepts.

This module provides ipywidgets-based interactive controls for
learning rate, convexity, bias-variance, and regularization.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, fixed
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def learning_rate_widget():
    """
    Interactive widget to explore learning rate effects on gradient descent.
    """
    def plot_gd(learning_rate, max_iters):
        from .optimizers import gradient_descent, compute_gradient
        from .loss_functions import quadratic_bowl
        
        # Simple quadratic loss
        loss_fn = quadratic_bowl
        grad_fn = lambda theta: compute_gradient(loss_fn, theta)
        
        # Initial point
        theta_init = np.array([3.0, 3.0])
        
        # Run GD
        result = gradient_descent(loss_fn, grad_fn, theta_init, 
                                 learning_rate=learning_rate, max_iters=max_iters)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss history
        ax1.plot(result['loss_history'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Loss Convergence (η={learning_rate})', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Trajectory
        from .visualizations import plot_loss_landscape_2d
        _, ax2, _, _, _ = plot_loss_landscape_2d(loss_fn, (-4, 4), (-4, 4), 
                                                  resolution=50, title='Parameter Trajectory')
        ax2.plot(result['theta_history'][:, 0], result['theta_history'][:, 1], 
                'ro-', linewidth=2, markersize=6, label='GD path')
        ax2.plot(result['theta_history'][0, 0], result['theta_history'][0, 1], 
                'g*', markersize=15, label='Start')
        ax2.plot(result['theta_history'][-1, 0], result['theta_history'][-1, 1], 
                'r*', markersize=15, label='End')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final loss: {result['loss']:.6f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Final θ: [{result['theta'][0]:.4f}, {result['theta'][1]:.4f}]")
    
    interact(plot_gd,
            learning_rate=FloatSlider(min=0.01, max=1.0, step=0.01, value=0.1,
                                     description='Learning Rate (η):',
                                     style={'description_width': 'initial'}),
            max_iters=IntSlider(min=10, max=100, step=10, value=50,
                               description='Max Iterations:',
                               style={'description_width': 'initial'}))


def gd_comparison_widget():
    """
    Interactive widget to compare Batch GD, SGD, and Minibatch GD.
    """
    def plot_comparison(batch_size, n_samples):
        from .ce_examples import create_regression_data
        from .loss_functions import mse_loss, mse_gradient
        from .optimizers import gradient_descent, stochastic_gradient_descent
        
        # Generate data
        data = create_regression_data(n_samples=n_samples, n_features=1, 
                                     noise_std=2.0, seed=42)
        X, y = data['X'], data['y']
        
        # Initial parameters
        theta_init = np.zeros(X.shape[1])
        
        # Batch GD
        loss_fn_batch = lambda theta: mse_loss(theta, X, y)
        grad_fn_batch = lambda theta: mse_gradient(theta, X, y)
        result_batch = gradient_descent(loss_fn_batch, grad_fn_batch, theta_init,
                                       learning_rate=0.01, max_iters=50)
        
        # SGD (minibatch)
        loss_fn_sgd = mse_loss
        grad_fn_sgd = mse_gradient
        result_sgd = stochastic_gradient_descent(loss_fn_sgd, grad_fn_sgd, theta_init,
                                                X, y, batch_size=batch_size,
                                                learning_rate=0.01, max_epochs=50)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(result_batch['loss_history'], 'b-', linewidth=2, label='Batch GD')
        ax.plot(result_sgd['loss_history'], 'r-', linewidth=1, alpha=0.7, label=f'Minibatch GD (size={batch_size})')
        
        ax.set_xlabel('Iteration/Update', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.set_title('Batch GD vs Minibatch GD', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Batch GD - Final loss: {result_batch['loss']:.4f}")
        print(f"Minibatch GD - Final loss: {result_sgd['loss']:.4f}")
    
    interact(plot_comparison,
            batch_size=IntSlider(min=1, max=50, step=1, value=16,
                                description='Batch Size:',
                                style={'description_width': 'initial'}),
            n_samples=IntSlider(min=50, max=200, step=50, value=100,
                               description='Dataset Size:',
                               style={'description_width': 'initial'}))


def convexity_widget():
    """
    Interactive widget to explore convex and non-convex functions.
    """
    def plot_function(function_type, show_gradient):
        from .loss_functions import quadratic_bowl, rosenbrock, himmelblau
        from .visualizations import plot_loss_landscape_2d, plot_gradient_field
        from .optimizers import compute_gradient
        
        # Select function
        if function_type == 'Quadratic (Convex)':
            loss_fn = quadratic_bowl
            x_range, y_range = (-3, 3), (-3, 3)
        elif function_type == 'Rosenbrock (Non-convex)':
            loss_fn = rosenbrock
            x_range, y_range = (-2, 2), (-1, 3)
        elif function_type == 'Himmelblau (Multi-modal)':
            loss_fn = himmelblau
            x_range, y_range = (-5, 5), (-5, 5)
        
        if show_gradient:
            grad_fn = lambda theta: compute_gradient(loss_fn, theta)
            fig, ax = plot_gradient_field(loss_fn, grad_fn, x_range, y_range,
                                         resolution=15, title=function_type)
        else:
            fig, ax, _, _, _ = plot_loss_landscape_2d(loss_fn, x_range, y_range,
                                                       title=function_type)
        
        plt.show()
    
    interact(plot_function,
            function_type=Dropdown(options=['Quadratic (Convex)', 
                                           'Rosenbrock (Non-convex)',
                                           'Himmelblau (Multi-modal)'],
                                  value='Quadratic (Convex)',
                                  description='Function:',
                                  style={'description_width': 'initial'}),
            show_gradient=Dropdown(options=[True, False], value=False,
                                  description='Show Gradients:',
                                  style={'description_width': 'initial'}))


def bias_variance_widget():
    """
    Interactive widget to explore bias-variance tradeoff.
    """
    def plot_tradeoff(n_samples, noise_std):
        from .ce_examples import simulate_overfitting
        from .visualizations import plot_bias_variance_tradeoff
        
        # Simulate overfitting
        result = simulate_overfitting(n_samples=n_samples, degree_range=(1, 15),
                                     noise_std=noise_std, seed=42)
        
        # Find optimal degree (minimum test error)
        optimal_idx = np.argmin(result['test_errors'])
        optimal_degree = result['degrees'][optimal_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bias-variance curve
        ax1.plot(result['degrees'], result['train_errors'], 'bo-', 
                linewidth=2, markersize=8, label='Training Error')
        ax1.plot(result['degrees'], result['test_errors'], 'ro-', 
                linewidth=2, markersize=8, label='Test Error')
        ax1.axvline(optimal_degree, color='g', linestyle='--', linewidth=2,
                   label=f'Optimal (degree={optimal_degree})')
        ax1.set_xlabel('Polynomial Degree', fontsize=12)
        ax1.set_ylabel('Error (MSE)', fontsize=12)
        ax1.set_title('Bias-Variance Tradeoff', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Show fit for optimal degree
        X_train = result['X_train']
        y_train = result['y_train']
        X_test = result['X_test']
        y_test = result['y_test']
        theta = result['models'][optimal_idx]
        
        X_test_poly = np.column_stack([X_test**i for i in range(optimal_degree + 1)])
        y_pred = X_test_poly @ theta
        
        ax2.scatter(X_train, y_train, s=50, alpha=0.6, label='Training data')
        ax2.plot(X_test, y_test, 'g-', linewidth=2, label='True function')
        ax2.plot(X_test, y_pred, 'r--', linewidth=2, label=f'Fit (degree={optimal_degree})')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title('Optimal Model Fit', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Optimal degree: {optimal_degree}")
        print(f"Test error at optimal: {result['test_errors'][optimal_idx]:.4f}")
    
    interact(plot_tradeoff,
            n_samples=IntSlider(min=10, max=50, step=10, value=20,
                               description='Training Samples:',
                               style={'description_width': 'initial'}),
            noise_std=FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5,
                                 description='Noise Level:',
                                 style={'description_width': 'initial'}))


def regularization_widget():
    """
    Interactive widget to explore L1 vs L2 regularization.
    """
    def plot_regularization(lambda_reg, reg_type):
        from .ce_examples import create_regression_data
        from .loss_functions import (mse_with_l1_reg, mse_with_l1_gradient,
                                     mse_with_l2_reg, mse_with_l2_gradient,
                                     mse_loss)
        from .optimizers import gradient_descent
        
        # Generate data with many features
        data = create_regression_data(n_samples=50, n_features=10, 
                                     noise_std=1.0, seed=42)
        X, y = data['X'], data['y']
        
        # Fit without regularization
        loss_fn_no_reg = lambda theta: mse_loss(theta, X, y)
        from .optimizers import compute_gradient
        grad_fn_no_reg = lambda theta: compute_gradient(loss_fn_no_reg, theta)
        
        theta_init = np.zeros(X.shape[1])
        result_no_reg = gradient_descent(loss_fn_no_reg, grad_fn_no_reg, theta_init,
                                        learning_rate=0.01, max_iters=100)
        
        # Fit with regularization
        if reg_type == 'L1 (LASSO)':
            loss_fn_reg = lambda theta: mse_with_l1_reg(theta, X, y, lambda_reg)
            grad_fn_reg = lambda theta: mse_with_l1_gradient(theta, X, y, lambda_reg)
        else:  # L2 (Ridge)
            loss_fn_reg = lambda theta: mse_with_l2_reg(theta, X, y, lambda_reg)
            grad_fn_reg = lambda theta: mse_with_l2_gradient(theta, X, y, lambda_reg)
        
        result_reg = gradient_descent(loss_fn_reg, grad_fn_reg, theta_init,
                                     learning_rate=0.01, max_iters=100)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Weight comparison
        feature_indices = range(len(result_no_reg['theta']))
        ax1.bar([i - 0.2 for i in feature_indices], result_no_reg['theta'], 
               width=0.4, label='No Regularization', alpha=0.7)
        ax1.bar([i + 0.2 for i in feature_indices], result_reg['theta'], 
               width=0.4, label=f'{reg_type} (λ={lambda_reg})', alpha=0.7)
        ax1.set_xlabel('Feature Index', fontsize=12)
        ax1.set_ylabel('Weight Value', fontsize=12)
        ax1.set_title('Weight Comparison', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Loss history
        ax2.plot(result_no_reg['loss_history'], 'b-', linewidth=2, label='No Regularization')
        ax2.plot(result_reg['loss_history'], 'r-', linewidth=2, label=f'{reg_type}')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Loss Convergence', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        print(f"No regularization - Final loss: {result_no_reg['loss']:.4f}")
        print(f"{reg_type} - Final loss: {result_reg['loss']:.4f}")
        print(f"Weight sparsity (|θ| < 0.1): {np.sum(np.abs(result_reg['theta']) < 0.1)}/{len(result_reg['theta'])}")
    
    interact(plot_regularization,
            lambda_reg=FloatSlider(min=0.0, max=2.0, step=0.1, value=0.5,
                                  description='λ (Regularization):',
                                  style={'description_width': 'initial'}),
            reg_type=Dropdown(options=['L1 (LASSO)', 'L2 (Ridge)'],
                             value='L2 (Ridge)',
                             description='Regularization Type:',
                             style={'description_width': 'initial'}))
