"""
Similarity and Distance Metrics

This module provides various distance and similarity metrics
for clustering algorithms.
"""

import numpy as np


def euclidean_distance(x, y):
    """
    Euclidean (L2) distance.
    
    d(x, y) = sqrt(Σ (x_i - y_i)²)
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    
    Returns:
    --------
    float
        Euclidean distance
    
    Notes:
    ------
    - Most common distance metric
    - Sensitive to scale (normalize features first)
    - Used in K-Means clustering
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x - y)**2))


def manhattan_distance(x, y):
    """
    Manhattan (L1) distance.
    
    d(x, y) = Σ |x_i - y_i|
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    
    Returns:
    --------
    float
        Manhattan distance
    
    Notes:
    ------
    - Also called "city block" or "taxicab" distance
    - Movement along axes only
    - More robust to outliers than Euclidean
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(np.abs(x - y))


def minkowski_distance(x, y, p=2):
    """
    Minkowski distance (generalized Lp norm).
    
    d(x, y) = (Σ |x_i - y_i|^p)^(1/p)
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    p : float
        Order of the norm
        - p=1: Manhattan distance
        - p=2: Euclidean distance
        - p→∞: Chebyshev distance
    
    Returns:
    --------
    float
        Minkowski distance
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.power(np.sum(np.abs(x - y)**p), 1/p)


def cosine_similarity(x, y):
    """
    Cosine similarity.
    
    sim(x, y) = (x · y) / (||x|| ||y||)
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    
    Returns:
    --------
    float
        Cosine similarity in range [-1, 1]
        - 1: identical direction
        - 0: orthogonal
        - -1: opposite direction
    
    Notes:
    ------
    - Measures angle between vectors, not magnitude
    - Common in text mining and high-dimensional data
    - Does not require normalization
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    norm_x = np.sqrt(np.sum(x**2))
    norm_y = np.sqrt(np.sum(y**2))
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    return np.dot(x, y) / (norm_x * norm_y)


def cosine_distance(x, y):
    """
    Cosine distance.
    
    d(x, y) = 1 - cosine_similarity(x, y)
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    
    Returns:
    --------
    float
        Cosine distance in range [0, 2]
    """
    return 1 - cosine_similarity(x, y)


def hamming_distance(x, y):
    """
    Hamming distance for categorical/binary data.
    
    d(x, y) = number of positions where x_i ≠ y_i
    
    Parameters:
    -----------
    x, y : array_like
        Binary or categorical vectors
    
    Returns:
    --------
    int
        Hamming distance
    
    Notes:
    ------
    - Used for binary vectors, strings, DNA sequences
    - Counts mismatches
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(x != y)


def jaccard_similarity(x, y):
    """
    Jaccard similarity for sets.
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Parameters:
    -----------
    x, y : array_like
        Binary vectors (1 = present, 0 = absent)
        or sets
    
    Returns:
    --------
    float
        Jaccard similarity in range [0, 1]
        - 1: identical sets
        - 0: no overlap
    
    Notes:
    ------
    - Used for document similarity, recommendation systems
    - For binary vectors: treats 1s as set membership
    """
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)
    
    intersection = np.sum(x & y)
    union = np.sum(x | y)
    
    if union == 0:
        return 1.0  # Both empty sets are identical
    
    return intersection / union


def jaccard_distance(x, y):
    """
    Jaccard distance.
    
    d(x, y) = 1 - Jaccard_similarity(x, y)
    """
    return 1 - jaccard_similarity(x, y)


def pairwise_distances(X, metric='euclidean', **kwargs):
    """
    Compute pairwise distance matrix.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    metric : str or callable
        Distance metric: 'euclidean', 'manhattan', 'cosine', 
        'minkowski', 'hamming', 'jaccard'
    **kwargs : dict
        Additional arguments for metric (e.g., p for minkowski)
    
    Returns:
    --------
    array_like, shape (n_samples, n_samples)
        Symmetric distance matrix
    """
    n_samples = X.shape[0]
    
    # Select metric function
    if callable(metric):
        metric_fn = metric
    else:
        metric_map = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': cosine_distance,
            'hamming': hamming_distance,
            'jaccard': jaccard_distance,
        }
        
        if metric == 'minkowski':
            p = kwargs.get('p', 2)
            metric_fn = lambda x, y: minkowski_distance(x, y, p=p)
        elif metric in metric_map:
            metric_fn = metric_map[metric]
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Compute distances
    D = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = metric_fn(X[i], X[j])
            D[i, j] = dist
            D[j, i] = dist
    
    return D


def compare_metrics(x, y):
    """
    Compare all metrics for two points.
    
    Parameters:
    -----------
    x, y : array_like
        Data points
    
    Returns:
    --------
    dict
        All metric values
    """
    return {
        'Euclidean': euclidean_distance(x, y),
        'Manhattan': manhattan_distance(x, y),
        'Cosine Similarity': cosine_similarity(x, y),
        'Cosine Distance': cosine_distance(x, y),
        'Minkowski (p=3)': minkowski_distance(x, y, p=3)
    }
