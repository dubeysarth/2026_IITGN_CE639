"""
Interpretability Analysis for Dimensionality Reduction

This module provides tools to analyze and visualize the interpretability
of principal components and compressed features.
"""

import numpy as np
import matplotlib.pyplot as plt


def pca_loadings_plot(pca_result, feature_names=None, n_components=None, figsize=(12, 6)):
    """
    Plot PCA loadings (component coefficients).
    
    Loadings show how each original feature contributes to each PC.
    
    Parameters:
    -----------
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    feature_names : list, optional
        Names of original features
    n_components : int, optional
        Number of components to plot
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    
    Notes:
    ------
    High absolute loading = feature strongly contributes to PC
    """
    components = pca_result['components']
    
    if n_components is None:
        n_components = min(5, components.shape[0])
    
    n_features = components.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_features)
    width = 0.8 / n_components
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_components))
    
    for i in range(n_components):
        offset = (i - n_components/2) * width
        ax.bar(x + offset, components[i], width, label=f'PC{i+1}', 
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Original Features', fontsize=11)
    ax.set_ylabel('Loading (Coefficient)', fontsize=11)
    ax.set_title('PCA Loadings: Feature Contributions to PCs', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    return fig, ax


def feature_importance_from_pca(pca_result, feature_names=None):
    """
    Compute feature importance scores from PCA.
    
    Importance = weighted sum of absolute loadings across PCs,
    weighted by explained variance.
    
    Parameters:
    -----------
    pca_result : dict
        Result from pca_covariance() or pca_svd()
    feature_names : list, optional
        Names of features
    
    Returns:
    --------
    dict
        Contains:
        - 'importance': Importance score per feature
        - 'feature_names': Feature names
        - 'ranking': Features sorted by importance
    
    Notes:
    ------
    This provides a rough measure of which original features
    are most important in the compressed representation.
    """
    components = pca_result['components']
    explained_var_ratio = pca_result['explained_variance_ratio']
    
    n_features = components.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Weighted sum of absolute loadings
    importance = np.zeros(n_features)
    
    for i, (comp, var_ratio) in enumerate(zip(components, explained_var_ratio)):
        importance += np.abs(comp) * var_ratio
    
    # Normalize
    importance = importance / np.sum(importance)
    
    # Ranking
    ranking_indices = np.argsort(importance)[::-1]
    ranking = [(feature_names[i], importance[i]) for i in ranking_indices]
    
    return {
        'importance': importance,
        'feature_names': feature_names,
        'ranking': ranking
    }


def compare_original_vs_compressed(X, pca_result, sample_idx=0, feature_names=None, 
                                   figsize=(14, 5)):
    """
    Compare original features vs PC representation for a sample.
    
    Parameters:
    -----------
    X : array_like
        Original data
    pca_result : dict
        PCA result
    sample_idx : int
        Index of sample to visualize
    feature_names : list, optional
        Feature names
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .pca import transform_pca, inverse_transform_pca
    
    n_features = X.shape[1]
    n_components = pca_result['n_components']
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Get sample
    x_original = X[sample_idx]
    
    # Transform to PC space
    Z_PC = transform_pca(X[sample_idx:sample_idx+1], pca_result)
    z_pc = Z_PC[0]
    
    # Reconstruct
    X_reconstructed = inverse_transform_pca(Z_PC, pca_result)
    x_reconstructed = X_reconstructed[0]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original features
    ax = axes[0]
    ax.bar(range(n_features), x_original, color='steelblue', edgecolor='black')
    ax.set_xlabel('Feature Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Original Features', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # PC representation
    ax = axes[1]
    ax.bar(range(n_components), z_pc, color='coral', edgecolor='black')
    ax.set_xlabel('PC Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title(f'Compressed ({n_components} PCs)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Reconstructed features
    ax = axes[2]
    width = 0.35
    x_pos = np.arange(n_features)
    ax.bar(x_pos - width/2, x_original, width, label='Original', 
           color='steelblue', edgecolor='black')
    ax.bar(x_pos + width/2, x_reconstructed, width, label='Reconstructed', 
           color='lightcoral', edgecolor='black')
    ax.set_xlabel('Feature Index', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Original vs Reconstructed', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Reconstruction error
    mse = np.mean((x_original - x_reconstructed) ** 2)
    fig.suptitle(f'Sample {sample_idx} - Reconstruction MSE: {mse:.4f}', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig, axes


def interpretability_score(pca_result, threshold=0.8):
    """
    Compute an interpretability score for PCA.
    
    A PC is "interpretable" if dominated by few features.
    
    Parameters:
    -----------
    pca_result : dict
        PCA result
    threshold : float
        Cumulative loading threshold for interpretability
    
    Returns:
    --------
    dict
        Contains:
        - 'scores': Interpretability score per PC (0-1)
        - 'dominant_features': Number of features needed to explain threshold
    
    Notes:
    ------
    Lower score = more interpretable (fewer features dominate)
    """
    components = pca_result['components']
    n_components, n_features = components.shape
    
    scores = []
    dominant_counts = []
    
    for comp in components:
        # Sort by absolute loading
        abs_loadings = np.abs(comp)
        sorted_loadings = np.sort(abs_loadings)[::-1]
        
        # Cumulative sum
        cumsum = np.cumsum(sorted_loadings) / np.sum(abs_loadings)
        
        # Number of features to reach threshold
        n_dominant = np.searchsorted(cumsum, threshold) + 1
        
        # Score: fraction of features needed
        score = n_dominant / n_features
        
        scores.append(score)
        dominant_counts.append(n_dominant)
    
    return {
        'scores': np.array(scores),
        'dominant_features': np.array(dominant_counts),
        'threshold': threshold
    }
