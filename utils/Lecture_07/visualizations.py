"""
General Visualization Helpers

This module provides visualization functions for loss landscapes,
training history, threshold impacts, and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid_vs_linear(figsize=(12, 5)):
    """
    Compare sigmoid activation to linear function.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from .logistic_regression import sigmoid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    z = np.linspace(-10, 10, 200)
    
    # Left: Sigmoid vs Linear
    ax1.plot(z, sigmoid(z), linewidth=3, color='#2E86AB', label='Sigmoid σ(z)')
    ax1.plot(z, z, linewidth=3, color='#A23B72', linestyle='--', label='Linear z')
    ax1.axhline(0.5, color='red', linestyle=':', alpha=0.5)
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax1.axhline(1, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('z', fontsize=12)
    ax1.set_ylabel('Output', fontsize=12)
    ax1.set_title('Sigmoid vs Linear', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1, 2])
    
    # Right: Sigmoid derivative
    sigma_z = sigmoid(z)
    derivative = sigma_z * (1 - sigma_z)
    
    ax2.plot(z, derivative, linewidth=3, color='#F18F01')
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('z', fontsize=12)
    ax2.set_ylabel("σ'(z) = σ(z)(1-σ(z))", fontsize=12)
    ax2.set_title('Sigmoid Derivative', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_softmax_demo(z_values=None, figsize=(10, 6)):
    """
    Demonstrate softmax function for multi-class classification.
    
    Parameters:
    -----------
    z_values : array_like, optional
        Logits for K classes. If None, uses example values.
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    from .logistic_regression import softmax
    
    if z_values is None:
        z_values = np.array([2.0, 1.0, 0.5])
    
    # Compute softmax
    probs = softmax(z_values.reshape(1, -1))[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    classes = [f'Class {i}' for i in range(len(z_values))]
    x_pos = np.arange(len(classes))
    
    # Left: Logits
    bars1 = ax1.bar(x_pos, z_values, color='#2E86AB', alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Logit (z)', fontsize=12)
    ax1.set_title('Input Logits', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(classes)
    ax1.grid(axis='y', alpha=0.3)
    
    # Annotate
    for bar, val in zip(bars1, z_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Right: Probabilities
    bars2 = ax2.bar(x_pos, probs, color='#A23B72', alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Softmax Probabilities', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(classes)
    ax2.set_ylim([0, 1])
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Annotate
    for bar, val in zip(bars2, probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add sum annotation
    ax2.text(0.5, 0.95, f'Sum = {probs.sum():.3f}', 
            transform=ax2.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11)
    
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_loss_landscape_classification(X, y, w_range=(-3, 3), b_range=(-3, 3),
                                      resolution=50, figsize=(12, 5)):
    """
    Plot cross-entropy loss landscape for simple 1D logistic regression.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 1)
        1D features
    y : array_like, shape (n_samples,)
        Binary labels
    w_range : tuple
        Range for weight parameter
    b_range : tuple
        Range for bias parameter
    resolution : int
        Grid resolution
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from .logistic_regression import sigmoid, cross_entropy_loss
    
    # Create parameter grid
    w_vals = np.linspace(w_range[0], w_range[1], resolution)
    b_vals = np.linspace(b_range[0], b_range[1], resolution)
    W, B = np.meshgrid(w_vals, b_vals)
    
    # Compute loss for each parameter combination
    Loss = np.zeros_like(W)
    for i in range(resolution):
        for j in range(resolution):
            w, b = W[i, j], B[i, j]
            z = X.flatten() * w + b
            y_pred = sigmoid(z)
            Loss[i, j] = cross_entropy_loss(y, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Contour plot
    contour = ax1.contourf(W, B, Loss, levels=20, cmap='viridis')
    ax1.contour(W, B, Loss, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    fig.colorbar(contour, ax=ax1, label='Loss')
    
    # Mark minimum
    min_idx = np.unravel_index(np.argmin(Loss), Loss.shape)
    ax1.plot(W[min_idx], B[min_idx], 'r*', markersize=20, 
            label=f'Min: w={W[min_idx]:.2f}, b={B[min_idx]:.2f}')
    
    ax1.set_xlabel('Weight (w)', fontsize=12)
    ax1.set_ylabel('Bias (b)', fontsize=12)
    ax1.set_title('Loss Landscape (Contour)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(W, B, Loss, cmap='viridis', alpha=0.8, 
                           edgecolor='none', antialiased=True)
    ax2.set_xlabel('Weight (w)', fontsize=11)
    ax2.set_ylabel('Bias (b)', fontsize=11)
    ax2.set_zlabel('Loss', fontsize=11)
    ax2.set_title('Loss Landscape (3D)', fontsize=13, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_training_history(history, figsize=(14, 5)):
    """
    Plot training loss and accuracy history.
    
    Parameters:
    -----------
    history : dict
        Must contain 'loss_history' and optionally 'accuracy_history'
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    has_accuracy = 'accuracy_history' in history
    
    if has_accuracy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
    
    # Plot loss
    iterations = np.arange(len(history['loss_history']))
    ax1.plot(iterations, history['loss_history'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate final loss
    final_loss = history['loss_history'][-1]
    ax1.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}',
            transform=ax1.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10)
    
    # Plot accuracy if available
    if has_accuracy:
        ax2.plot(iterations, history['accuracy_history'], linewidth=2, color='#A23B72')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Annotate final accuracy
        final_acc = history['accuracy_history'][-1]
        ax2.text(0.95, 0.05, f'Final Accuracy: {final_acc:.4f}',
                transform=ax2.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=10)
    
    fig.tight_layout()
    return fig, (ax1, ax2) if has_accuracy else (fig, ax1)


def plot_threshold_impact(y_true, y_scores, thresholds=None, figsize=(14, 10)):
    """
    Show impact of threshold on classification metrics.
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_scores : array_like
        Predicted probabilities
    thresholds : array_like, optional
        Thresholds to evaluate. If None, uses [0.3, 0.5, 0.7]
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    from .metrics import precision, recall, f1_score, accuracy
    
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Compute metrics for range of thresholds
    threshold_range = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in threshold_range:
        y_pred = (y_scores >= thresh).astype(int)
        accuracies.append(accuracy(y_true, y_pred))
        precisions.append(precision(y_true, y_pred))
        recalls.append(recall(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Plot metrics vs threshold
    ax = axes[0]
    ax.plot(threshold_range, accuracies, linewidth=2, label='Accuracy', color='#2E86AB')
    ax.plot(threshold_range, precisions, linewidth=2, label='Precision', color='#A23B72')
    ax.plot(threshold_range, recalls, linewidth=2, label='Recall', color='#F18F01')
    ax.plot(threshold_range, f1_scores, linewidth=2, label='F1 Score', color='#06A77D')
    
    # Mark selected thresholds
    for thresh in thresholds:
        ax.axvline(thresh, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Metrics vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot confusion matrices for selected thresholds
    from .metrics import confusion_matrix, plot_confusion_matrix
    
    for idx, thresh in enumerate(thresholds[:3]):
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        ax = axes[idx + 1]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Pred 0', 'Pred 1'],
               yticklabels=['True 0', 'True 1'],
               ylabel='True Label',
               xlabel='Predicted Label',
               title=f'Threshold = {thresh:.1f}')
        
        # Annotate
        thresh_val = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh_val else "black",
                       fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig, axes


def plot_class_separation(X, y, feature_names=None, figsize=(12, 5)):
    """
    Visualize class separation in feature space.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Features (works best with 1-3 features)
    y : array_like, shape (n_samples,)
        Binary labels
    feature_names : list, optional
        Feature names
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    n_features = X.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    if n_features == 1:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram
        ax.hist(X[y == 0], bins=20, alpha=0.6, label='Class 0', color='#2E86AB')
        ax.hist(X[y == 1], bins=20, alpha=0.6, label='Class 1', color='#A23B72')
        ax.set_xlabel(feature_names[0], fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    elif n_features == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='#2E86AB', s=60, 
                   alpha=0.7, edgecolors='black', linewidths=1, label='Class 0')
        ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='#A23B72', s=60, 
                   alpha=0.7, edgecolors='black', linewidths=1, label='Class 1')
        ax1.set_xlabel(feature_names[0], fontsize=12)
        ax1.set_ylabel(feature_names[1], fontsize=12)
        ax1.set_title('Feature Space', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Marginal distributions
        ax2.hist(X[y == 0, 0], bins=15, alpha=0.6, label=f'{feature_names[0]} (Class 0)', 
                color='#2E86AB')
        ax2.hist(X[y == 1, 0], bins=15, alpha=0.6, label=f'{feature_names[0]} (Class 1)', 
                color='#A23B72')
        ax2.set_xlabel('Feature Value', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Marginal Distributions', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        return fig, (ax1, ax2)
    
    else:
        # For higher dimensions, show pairwise scatter
        print("Note: Showing first 2 features only for visualization")
        return plot_class_separation(X[:, :2], y, feature_names[:2], figsize)
