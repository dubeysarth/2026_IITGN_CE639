"""
Visualizations for Dimensionality Reduction

This module provides plotting functions for PCA, autoencoders,
and curse of dimensionality demonstrations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_scree(pca_result, threshold=0.9, figsize=(12, 5)):
    """
    Plot scree plot showing explained variance.
    
    Parameters:
    -----------
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    threshold : float
        Cumulative variance threshold to highlight
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    var_ratio = pca_result['explained_variance_ratio']
    cumsum_var = np.cumsum(var_ratio)
    n_components = len(var_ratio)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax = axes[0]
    ax.bar(range(1, n_components + 1), var_ratio, color='steelblue', 
           edgecolor='black', alpha=0.7)
    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Explained Variance Ratio', fontsize=11)
    ax.set_title('Scree Plot: Individual Variance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative variance
    ax = axes[1]
    ax.plot(range(1, n_components + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, 
              label=f'{threshold*100:.0f}% threshold')
    
    # Find number of components for threshold
    n_for_threshold = np.searchsorted(cumsum_var, threshold) + 1
    # Clamp to available components
    n_for_threshold = min(n_for_threshold, n_components)
    
    ax.axvline(n_for_threshold, color='green', linestyle='--', linewidth=2,
              label=f'{n_for_threshold} components')
    ax.scatter([n_for_threshold], [cumsum_var[n_for_threshold-1]], 
              c='green', s=200, marker='*', zorder=10)
    
    ax.set_xlabel('Number of Components', fontsize=11)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=11)
    ax.set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes


def plot_pca_2d(X, pca_result, labels=None, title='PCA 2D Projection', figsize=(10, 8)):
    """
    Plot 2D PCA projection.
    
    Parameters:
    -----------
    X : array_like
        Original data
    pca_result : dict
        PCA result
    labels : array_like, optional
        Class labels for coloring
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from .pca import transform_pca
    
    # Transform to PC space
    Z_PC = transform_pca(X, pca_result)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(Z_PC[mask, 0], Z_PC[mask, 1], c=[colors[i]], 
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                      label=f'Class {label}')
        ax.legend()
    else:
        ax.scatter(Z_PC[:, 0], Z_PC[:, 1], c='steelblue', s=60, 
                  alpha=0.7, edgecolors='black', linewidth=0.5)
    
    var1 = pca_result['explained_variance_ratio'][0]
    var2 = pca_result['explained_variance_ratio'][1]
    
    ax.set_xlabel(f'PC1 ({var1*100:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2*100:.1f}% variance)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    return fig, ax


def plot_pca_3d(X, pca_result, labels=None, title='PCA 3D Projection', figsize=(12, 9)):
    """
    Plot 3D PCA projection.
    
    Parameters:
    -----------
    X : array_like
        Original data
    pca_result : dict
        PCA result (must have at least 3 components)
    labels : array_like, optional
        Class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    from .pca import transform_pca
    
    Z_PC = transform_pca(X, pca_result)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(Z_PC[mask, 0], Z_PC[mask, 1], Z_PC[mask, 2], 
                      c=[colors[i]], s=50, alpha=0.7, edgecolors='black', 
                      linewidths=0.3, label=f'Class {label}')
        ax.legend()
    else:
        ax.scatter(Z_PC[:, 0], Z_PC[:, 1], Z_PC[:, 2], c='steelblue', 
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.3)
    
    var1 = pca_result['explained_variance_ratio'][0]
    var2 = pca_result['explained_variance_ratio'][1]
    var3 = pca_result['explained_variance_ratio'][2]
    
    ax.set_xlabel(f'PC1 ({var1*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2*100:.1f}%)', fontsize=10)
    ax.set_zlabel(f'PC3 ({var3*100:.1f}%)', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    return fig, ax


def plot_autoencoder_reconstruction(X, autoencoder, n_samples=5, figsize=(14, 6)):
    """
    Plot original vs reconstructed samples from autoencoder.
    
    Parameters:
    -----------
    X : array_like
        Data
    autoencoder : LinearAutoencoder or Autoencoder
        Trained autoencoder
    n_samples : int
        Number of samples to show
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    indices = np.random.choice(len(X), n_samples, replace=False)
    X_samples = X[indices]
    
    X_reconstructed, _ = autoencoder.forward(X_samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        # Original
        ax = axes[0, i]
        ax.plot(X_samples[i], 'b-', linewidth=2)
        ax.set_title(f'Original {i+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Value', fontsize=10)
        
        # Reconstructed
        ax = axes[1, i]
        ax.plot(X_reconstructed[i], 'r-', linewidth=2)
        mse = np.mean((X_samples[i] - X_reconstructed[i]) ** 2)
        ax.set_title(f'Reconstructed (MSE: {mse:.3f})', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Value', fontsize=10)
        ax.set_xlabel('Feature', fontsize=9)
    
    fig.suptitle('Autoencoder Reconstruction', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig, axes


def plot_latent_space(X, autoencoder, labels=None, title='Latent Space', figsize=(10, 8)):
    """
    Plot 2D latent space representation.
    
    Parameters:
    -----------
    X : array_like
        Data
    autoencoder : Autoencoder
        Trained autoencoder (latent_dim should be 2)
    labels : array_like, optional
        Class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    Z = autoencoder.encode(X)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(Z[mask, 0], Z[mask, 1], c=[colors[i]], 
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                      label=f'Class {label}')
        ax.legend()
    else:
        ax.scatter(Z[:, 0], Z[:, 1], c='coral', s=60, 
                  alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Latent Dimension 1', fontsize=11)
    ax.set_ylabel('Latent Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    return fig, ax


def plot_distance_concentration(distance_result, figsize=(12, 5)):
    """
    Plot distance concentration demonstration.
    
    Parameters:
    -----------
    distance_result : dict
        Result from distance_concentration_demo()
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    dims = distance_result['dimensions']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Min/Max/Mean distances
    ax = axes[0]
    ax.plot(dims, distance_result['min_distances'], 'b-', linewidth=2, 
           marker='o', label='Min Distance')
    ax.plot(dims, distance_result['max_distances'], 'r-', linewidth=2, 
           marker='s', label='Max Distance')
    ax.plot(dims, distance_result['mean_distances'], 'g--', linewidth=2, 
           marker='^', label='Mean Distance')
    ax.set_xlabel('Dimensions', fontsize=11)
    ax.set_ylabel('Distance', fontsize=11)
    ax.set_title('Distance vs Dimensionality', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ratio
    ax = axes[1]
    ax.plot(dims, distance_result['ratio'], 'purple', linewidth=3, marker='o')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect concentration')
    ax.set_xlabel('Dimensions', fontsize=11)
    ax.set_ylabel('Min/Max Ratio', fontsize=11)
    ax.set_title('Distance Concentration (Ratio → 1)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes


def plot_loadings_heatmap(pca_result, feature_names=None, n_components=None, figsize=(10, 8)):
    """
    Plot heatmap of PCA loadings.
    
    Parameters:
    -----------
    pca_result : dict
        PCA result
    feature_names : list, optional
        Feature names
    n_components : int, optional
        Number of components to show
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    components = pca_result['components']
    
    if n_components is None:
        n_components = min(10, components.shape[0])
    
    n_features = components.shape[1]
    
    if feature_names is None:
        feature_names = [f'F{i+1}' for i in range(n_features)]
    
    loadings = components[:n_components]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(loadings, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(range(n_components))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_components)])
    
    ax.set_xlabel('Original Features', fontsize=11)
    ax.set_ylabel('Principal Components', fontsize=11)
    ax.set_title('PCA Loadings Heatmap', fontsize=13, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading', fontsize=10)
    
    # Add value annotations for small matrices
    if n_components <= 5 and n_features <= 10:
        for i in range(n_components):
            for j in range(n_features):
                text = ax.text(j, i, f'{loadings[i, j]:.2f}',
                             ha='center', va='center', fontsize=8,
                             color='white' if abs(loadings[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    
    return fig, ax
