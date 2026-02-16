"""
Clustering Visualizations

This module provides visualization functions for clusters,
dendrograms, elbow plots, silhouette plots, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(X, labels, centroids=None, title='Cluster Visualization',
                     figsize=(10, 8)):
    """
    Plot 2D cluster visualization.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D data points
    labels : array_like, shape (n_samples,)
        Cluster labels
    centroids : array_like, optional
        Cluster centroids
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                  s=60, alpha=0.7, edgecolors='black', linewidths=0.5,
                  label=f'Cluster {label}')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                  marker='X', s=200, edgecolors='black', linewidths=2,
                  label='Centroids', zorder=10)
    
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_clusters_3d(X, labels, centroids=None, title='3D Cluster Visualization',
                     figsize=(12, 9)):
    """
    Plot 3D cluster visualization.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 3)
        3D data points
    labels : array_like, shape (n_samples,)
        Cluster labels
    centroids : array_like, optional
        Cluster centroids
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], c=[colors[i]], 
                  s=50, alpha=0.7, edgecolors='black', linewidths=0.3,
                  label=f'Cluster {label}')
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                  label='Centroids')
    
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_zlabel('Feature 3', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    return fig, ax


def plot_dendrogram(linkage_matrix, labels=None, title='Dendrogram',
                   figsize=(14, 7), color_threshold=None):
    """
    Plot hierarchical clustering dendrogram.
    
    Parameters:
    -----------
    linkage_matrix : array_like, shape (n_samples-1, 4)
        Linkage matrix from hierarchical clustering
    labels : list, optional
        Labels for leaves
    title : str
        Plot title
    figsize : tuple
        Figure size
    color_threshold : float, optional
        Threshold for coloring
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    try:
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if use_scipy:
        scipy_dendrogram(linkage_matrix, ax=ax, labels=labels,
                        color_threshold=color_threshold,
                        leaf_rotation=45, leaf_font_size=9)
    else:
        # Simple dendrogram drawing (without scipy)
        n_merges = linkage_matrix.shape[0]
        n_samples = n_merges + 1
        
        # Just show merge heights
        heights = linkage_matrix[:, 2]
        ax.bar(range(len(heights)), heights, color='steelblue', alpha=0.7)
        ax.set_xlabel('Merge Step', fontsize=11)
        ax.set_ylabel('Linkage Distance', fontsize=11)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color='r', linestyle='--', 
                  linewidth=2, label=f'Cut at {color_threshold:.2f}')
        ax.legend()
    
    return fig, ax


def plot_elbow(k_values, wcss_values, optimal_k=None, title='Elbow Plot',
               figsize=(10, 6)):
    """
    Plot elbow/scree plot for K-Means.
    
    Parameters:
    -----------
    k_values : list
        K values
    wcss_values : list
        WCSS for each K
    optimal_k : int, optional
        Highlight optimal K
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=10)
    
    if optimal_k is not None:
        idx = k_values.index(optimal_k)
        ax.axvline(x=optimal_k, color='red', linestyle='--', 
                  linewidth=2, label=f'Optimal K = {optimal_k}')
        ax.scatter([optimal_k], [wcss_values[idx]], c='red', 
                  s=200, marker='*', zorder=10)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    if optimal_k is not None:
        ax.legend(fontsize=10)
    
    return fig, ax


def plot_silhouette(X, labels, title='Silhouette Analysis', figsize=(12, 7)):
    """
    Plot silhouette analysis for clustering.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    labels : array_like, shape (n_samples,)
        Cluster labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from .cluster_validation import silhouette_samples, silhouette_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    s_samples = silhouette_samples(X, labels)
    s_score = silhouette_score(X, labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    
    y_lower = 10
    
    for i, label in enumerate(unique_labels):
        cluster_silhouette = s_samples[labels == label]
        cluster_silhouette.sort()
        
        cluster_size = len(cluster_silhouette)
        y_upper = y_lower + cluster_size
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                        facecolor=colors[i], edgecolor='black', alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(label), fontsize=10)
        
        y_lower = y_upper + 10
    
    ax.axvline(x=s_score, color='red', linestyle='--', linewidth=2,
              label=f'Average Silhouette = {s_score:.3f}')
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim([-0.2, 1])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig, ax


def plot_kmeans_animation(X, centroids_history, labels_final, figsize=(10, 8)):
    """
    Create frames for K-Means animation.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D data points
    centroids_history : list of arrays
        Centroids at each iteration
    labels_final : array_like
        Final cluster labels
    figsize : tuple
        Figure size
    
    Returns:
    --------
    list of (fig, ax) tuples
        One for each frame
    """
    frames = []
    
    n_clusters = centroids_history[0].shape[0]
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
    
    for i, centroids in enumerate(centroids_history):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color by nearest centroid
        labels = np.zeros(len(X), dtype=int)
        for j in range(len(X)):
            dists = [np.sum((X[j] - c)**2) for c in centroids]
            labels[j] = np.argmin(dists)
        
        # Plot points
        for k in range(n_clusters):
            mask = (labels == k)
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], 
                      s=50, alpha=0.6, edgecolors='black', linewidths=0.3)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                  marker='X', s=200, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.set_title(f'K-Means Iteration {i}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        frames.append((fig, ax))
    
    return frames


def plot_linkage_comparison(X, labels_dict, title='Linkage Method Comparison',
                           figsize=(16, 4)):
    """
    Compare different linkage methods.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, 2)
        2D data points
    labels_dict : dict
        {method_name: labels} for each linkage method
    title : str
        Overall title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    n_methods = len(labels_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method, labels) in zip(axes, labels_dict.items()):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 3)))
        
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                      s=40, alpha=0.7, edgecolors='black', linewidths=0.3)
        
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_title(f'{method}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    return fig, axes
