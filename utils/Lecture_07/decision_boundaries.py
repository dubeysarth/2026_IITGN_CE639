"""
Decision Boundary Visualization

This module provides functions for visualizing decision boundaries,
sigmoid curves, and animated training progression.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


def create_mesh_grid(X, resolution=0.02, padding=0.5):
    """
    Create a mesh grid for decision boundary visualization.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D feature data
    resolution : float
        Grid resolution (smaller = finer)
    padding : float
        Padding around data bounds
    
    Returns:
    --------
    xx, yy : arrays
        Mesh grid coordinates
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    return xx, yy


def plot_decision_boundary_2d(X, y, model_predict_fn, title='Decision Boundary',
                              figsize=(10, 7), resolution=0.02):
    """
    Plot 2D decision boundary with data points.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D features
    y : array_like, shape (n_samples,)
        Binary labels (0 or 1)
    model_predict_fn : callable
        Function that takes X and returns predicted probabilities
        Signature: model_predict_fn(X) -> probabilities
    title : str
        Plot title
    figsize : tuple
        Figure size
    resolution : float
        Grid resolution
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    
    Notes:
    ------
    Visualizes decision boundary as contour where P(y=1) = 0.5
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X, resolution=resolution)
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_predict_fn(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary (contour at 0.5)
    contour = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5, 
               linestyles='--', label='Decision Boundary')
    
    # Plot data points
    scatter = ax.scatter(X[y == 0, 0], X[y == 0, 1], 
                        c='#2E86AB', s=80, edgecolors='black', 
                        linewidths=1.5, label='Class 0', alpha=0.9)
    scatter = ax.scatter(X[y == 1, 0], X[y == 1, 1], 
                        c='#A23B72', s=80, edgecolors='black', 
                        linewidths=1.5, label='Class 1', alpha=0.9)
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('P(y=1)', fontsize=12)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_sigmoid_curve(figsize=(10, 6)):
    """
    Plot sigmoid activation function.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    from .logistic_regression import sigmoid
    
    fig, ax = plt.subplots(figsize=figsize)
    
    z = np.linspace(-10, 10, 200)
    sigma_z = sigmoid(z)
    
    # Plot sigmoid
    ax.plot(z, sigma_z, linewidth=3, color='#2E86AB', label='σ(z) = 1/(1+e⁻ᶻ)')
    
    # Plot threshold line
    ax.axhline(0.5, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Decision Threshold (0.5)')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Annotate key points
    ax.plot(0, 0.5, 'ro', markersize=10, label='σ(0) = 0.5')
    ax.text(0.5, 0.5, '  (0, 0.5)', fontsize=11, va='bottom')
    
    # Asymptotes
    ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('z = w^T x + b', fontsize=12)
    ax.set_ylabel('σ(z)', fontsize=12)
    ax.set_title('Sigmoid Activation Function', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    
    return fig, ax


def animate_decision_boundary(X, y, training_history, n_frames=50, 
                              interval=100, figsize=(10, 7)):
    """
    Create animation showing decision boundary evolution during training.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D features
    y : array_like, shape (n_samples,)
        Binary labels
    training_history : dict
        Must contain:
        - 'weights_history': List of weight arrays at each iteration
        - 'loss_history': List of loss values
    n_frames : int
        Number of frames in animation
    interval : int
        Milliseconds between frames
    figsize : tuple
        Figure size
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        Animation object
    
    Notes:
    ------
    Call plt.show() or save with anim.save('filename.mp4')
    """
    from .logistic_regression import logistic_predict
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Select frames to show
    n_iterations = len(training_history['weights_history'])
    frame_indices = np.linspace(0, n_iterations - 1, n_frames, dtype=int)
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X, resolution=0.02)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        
        iteration = frame_indices[frame_idx]
        weights = training_history['weights_history'][iteration]
        
        # Predict on grid
        Z = logistic_predict(grid_points, weights)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', 
                   linewidths=2.5, linestyles='--')
        
        # Plot data
        ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='#2E86AB', s=80, 
                   edgecolors='black', linewidths=1.5, alpha=0.9)
        ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='#A23B72', s=80, 
                   edgecolors='black', linewidths=1.5, alpha=0.9)
        
        ax1.set_xlabel('Feature 1', fontsize=12)
        ax1.set_ylabel('Feature 2', fontsize=12)
        ax1.set_title(f'Decision Boundary (Iteration {iteration})', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot loss history
        ax2.plot(training_history['loss_history'][:iteration+1], 
                linewidth=2, color='#2E86AB')
        ax2.scatter(iteration, training_history['loss_history'][iteration], 
                   s=100, c='red', zorder=5)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, n_iterations])
        ax2.set_ylim([0, max(training_history['loss_history']) * 1.1])
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, 
                        repeat=True, blit=False)
    
    return anim


def plot_multiclass_boundaries(X, y, model_predict_fn, class_names=None,
                               title='Multi-Class Decision Boundaries',
                               figsize=(10, 7), resolution=0.02):
    """
    Plot decision boundaries for multi-class classification.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D features
    y : array_like, shape (n_samples,)
        Class labels (0 to K-1)
    model_predict_fn : callable
        Function that takes X and returns predicted class labels
        Signature: model_predict_fn(X) -> class_labels
    class_names : list, optional
        Names for each class
    title : str
        Plot title
    figsize : tuple
        Figure size
    resolution : float
        Grid resolution
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X, resolution=resolution)
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_predict_fn(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Define colors for classes
    n_classes = len(np.unique(y))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    cmap = ListedColormap(colors)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=np.arange(n_classes + 1) - 0.5)
    
    # Plot data points
    for class_idx in range(n_classes):
        mask = (y == class_idx)
        label = class_names[class_idx] if class_names else f'Class {class_idx}'
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[class_idx]], 
                  s=80, edgecolors='black', linewidths=1.5, 
                  label=label, alpha=0.9)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_probability_surface_3d(X, y, model_predict_fn, figsize=(12, 8)):
    """
    Plot 3D probability surface for binary classification.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D features
    y : array_like, shape (n_samples,)
        Binary labels
    model_predict_fn : callable
        Function returning probabilities
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes (3D)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X, resolution=0.05)
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model_predict_fn(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot surface
    surf = ax.plot_surface(xx, yy, Z, cmap='RdYlBu_r', alpha=0.7, 
                          edgecolor='none', antialiased=True)
    
    # Plot decision boundary plane at z=0.5
    ax.plot_surface(xx, yy, np.full_like(Z, 0.5), alpha=0.2, color='gray')
    
    # Plot data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], 0, c='#2E86AB', s=50, 
              edgecolors='black', linewidths=1, alpha=0.8, label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], 1, c='#A23B72', s=50, 
              edgecolors='black', linewidths=1, alpha=0.8, label='Class 1')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_zlabel('P(y=1)', fontsize=11)
    ax.set_title('Probability Surface', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.view_init(elev=20, azim=45)
    
    return fig, ax
