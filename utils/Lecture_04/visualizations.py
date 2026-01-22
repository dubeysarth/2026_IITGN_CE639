"""
Visualization functions for optimization and loss landscapes.

This module provides plotting and animation utilities for gradient descent,
loss landscapes, and bias-variance tradeoff.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')


def plot_loss_landscape_2d(loss_fn, x_range, y_range, resolution=100, 
                           contour_levels=20, title="Loss Landscape"):
    """
    Plot 2D loss landscape as contour plot.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function f([x, y])
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Grid resolution
    contour_levels : int
        Number of contour levels
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax, X, Y, Z : matplotlib objects and grid data
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate loss on grid
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = loss_fn(np.array([X[i, j], Y[i, j]]))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot
    contour = ax.contour(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.6)
    contourf = ax.contourf(X, Y, Z, levels=contour_levels, cmap='viridis', alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label('Loss', rotation=270, labelpad=20)
    
    ax.set_xlabel('θ₁', fontsize=12)
    ax.set_ylabel('θ₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig, ax, X, Y, Z


def plot_loss_landscape_3d(loss_fn, x_range, y_range, resolution=50,
                           title="3D Loss Landscape"):
    """
    Plot 3D loss landscape as surface plot.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function f([x, y])
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Grid resolution
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib 3D objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate loss
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = loss_fn(np.array([X[i, j], Y[i, j]]))
    
    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          edgecolor='none', antialiased=True)
    
    ax.set_xlabel('θ₁', fontsize=11)
    ax.set_ylabel('θ₂', fontsize=11)
    ax.set_zlabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def animate_gradient_descent(loss_fn, theta_history, x_range, y_range,
                             resolution=100, interval=100, title="Gradient Descent"):
    """
    Animate gradient descent path on loss landscape.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function
    theta_history : array
        History of theta values (n_iters, 2)
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Grid resolution
    interval : int
        Animation interval (ms)
    title : str
        Plot title
    
    Returns:
    --------
    HTML animation object
    """
    # Create base landscape
    fig, ax, X, Y, Z = plot_loss_landscape_2d(loss_fn, x_range, y_range, 
                                               resolution, title=title)
    
    # Initialize path line and point
    line, = ax.plot([], [], 'r-', linewidth=2, label='GD path')
    point, = ax.plot([], [], 'ro', markersize=10, label='Current position')
    ax.legend()
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point
    
    def animate(frame):
        # Update path
        line.set_data(theta_history[:frame+1, 0], theta_history[:frame+1, 1])
        # Update current point
        point.set_data([theta_history[frame, 0]], [theta_history[frame, 1]])
        return line, point
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(theta_history), interval=interval,
                        blit=True, repeat=True)
    
    plt.close()
    return HTML(anim.to_jshtml())


def plot_learning_rate_comparison(loss_fn, grad_fn, theta_init, learning_rates,
                                  max_iters=50, figsize=(14, 5)):
    """
    Compare gradient descent with different learning rates.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function
    grad_fn : callable
        Gradient function
    theta_init : array
        Initial parameters
    learning_rates : list
        List of learning rates to compare
    max_iters : int
        Maximum iterations
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .optimizers import gradient_descent
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(learning_rates)))
    
    for i, lr in enumerate(learning_rates):
        result = gradient_descent(loss_fn, grad_fn, theta_init, 
                                 learning_rate=lr, max_iters=max_iters)
        
        # Plot loss history
        axes[0].plot(result['loss_history'], color=colors[i], 
                    linewidth=2, label=f'η = {lr}')
        
        # Plot parameter trajectory (if 2D)
        if len(theta_init) == 2:
            axes[1].plot(result['theta_history'][:, 0], 
                        result['theta_history'][:, 1],
                        'o-', color=colors[i], markersize=4, 
                        linewidth=1.5, label=f'η = {lr}')
    
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss vs Iteration', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    if len(theta_init) == 2:
        axes[1].set_xlabel('θ₁', fontsize=12)
        axes[1].set_ylabel('θ₂', fontsize=12)
        axes[1].set_title('Parameter Trajectory', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_convex_vs_nonconvex(figsize=(14, 5)):
    """
    Visualize convex vs non-convex functions.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    x = np.linspace(-3, 3, 200)
    
    # Convex: quadratic
    convex = x**2
    
    # Non-convex: multiple local minima
    nonconvex = x**4 - 4*x**2 + x
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convex
    axes[0].plot(x, convex, 'b-', linewidth=2)
    axes[0].fill_between(x, convex, alpha=0.3)
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].plot(0, 0, 'ro', markersize=10, label='Global minimum')
    axes[0].set_xlabel('θ', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Convex Function (Bowl-shaped)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Non-convex
    axes[1].plot(x, nonconvex, 'r-', linewidth=2)
    axes[1].fill_between(x, nonconvex, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Find local minima (approximately)
    local_min_indices = [40, 100, 160]
    for idx in local_min_indices:
        axes[1].plot(x[idx], nonconvex[idx], 'go', markersize=8)
    axes[1].plot([], [], 'go', markersize=8, label='Local minima')
    
    axes[1].set_xlabel('θ', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Non-Convex Function (Multiple Minima)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_bias_variance_tradeoff(degrees, train_errors, test_errors, 
                                optimal_degree=None, figsize=(10, 6)):
    """
    Plot bias-variance tradeoff U-curve.
    
    Parameters:
    -----------
    degrees : array-like
        Model complexity (e.g., polynomial degrees)
    train_errors : array-like
        Training errors
    test_errors : array-like
        Test errors
    optimal_degree : int, optional
        Optimal complexity to highlight
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(degrees, train_errors, 'bo-', linewidth=2, markersize=8, label='Training Error')
    ax.plot(degrees, test_errors, 'ro-', linewidth=2, markersize=8, label='Test Error')
    
    # Highlight optimal
    if optimal_degree is not None:
        idx = list(degrees).index(optimal_degree)
        ax.axvline(optimal_degree, color='g', linestyle='--', linewidth=2, 
                  label=f'Optimal (degree={optimal_degree})')
        ax.plot(optimal_degree, test_errors[idx], 'g*', markersize=20)
    
    # Annotate regions
    mid_idx = len(degrees) // 2
    ax.text(degrees[2], max(test_errors) * 0.9, 'High Bias\n(Underfitting)', 
           fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(degrees[-3], max(test_errors) * 0.9, 'High Variance\n(Overfitting)', 
           fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    ax.set_xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    ax.set_ylabel('Error (MSE)', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    return fig, ax


def plot_regularization_paths(lambdas, weights_l1, weights_l2, feature_names=None,
                              figsize=(14, 5)):
    """
    Plot regularization paths (weight shrinkage vs λ).
    
    Parameters:
    -----------
    lambdas : array-like
        Regularization strengths
    weights_l1 : array
        Weights for different λ with L1 (n_lambdas, n_features)
    weights_l2 : array
        Weights for different λ with L2
    feature_names : list, optional
        Feature names
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    n_features = weights_l1.shape[1]
    
    if feature_names is None:
        feature_names = [f'θ_{i}' for i in range(n_features)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_features))
    
    # L1 path
    for i in range(n_features):
        axes[0].plot(lambdas, weights_l1[:, i], color=colors[i], 
                    linewidth=2, label=feature_names[i])
    axes[0].set_xlabel('λ (Regularization Strength)', fontsize=12)
    axes[0].set_ylabel('Weight Value', fontsize=12)
    axes[0].set_title('L1 (LASSO) Regularization Path', fontsize=13, fontweight='bold')
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    # L2 path
    for i in range(n_features):
        axes[1].plot(lambdas, weights_l2[:, i], color=colors[i], 
                    linewidth=2, label=feature_names[i])
    axes[1].set_xlabel('λ (Regularization Strength)', fontsize=12)
    axes[1].set_ylabel('Weight Value', fontsize=12)
    axes[1].set_title('L2 (Ridge) Regularization Path', fontsize=13, fontweight='bold')
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    return fig, axes


def plot_gradient_field(loss_fn, grad_fn, x_range, y_range, resolution=20,
                       title="Gradient Field"):
    """
    Plot gradient vector field on loss landscape.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function
    grad_fn : callable
        Gradient function
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Arrow grid resolution
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    # Create contour plot
    fig, ax, X, Y, Z = plot_loss_landscape_2d(loss_fn, x_range, y_range, 
                                               resolution=100, title=title)
    
    # Create gradient field
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X_grad, Y_grad = np.meshgrid(x, y)
    
    U = np.zeros_like(X_grad)
    V = np.zeros_like(Y_grad)
    
    for i in range(resolution):
        for j in range(resolution):
            grad = grad_fn(np.array([X_grad[i, j], Y_grad[i, j]]))
            U[i, j] = -grad[0]  # Negative gradient (descent direction)
            V[i, j] = -grad[1]
    
    # Plot arrows
    ax.quiver(X_grad, Y_grad, U, V, color='red', alpha=0.6, 
             scale=50, width=0.003, headwidth=4)
    
    return fig, ax
