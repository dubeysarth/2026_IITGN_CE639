"""
Cluster Validation Metrics

This module provides elbow method, silhouette score, and other
metrics for determining optimal number of clusters.
"""

import numpy as np


def silhouette_samples(X, labels):
    """
    Compute silhouette coefficient for each sample.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    labels : array_like, shape (n_samples,)
        Cluster assignments
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Silhouette coefficient for each sample
    
    Notes:
    ------
    For each point i:
    - a_i = average distance to points in same cluster
    - b_i = min average distance to points in nearest other cluster
    - s_i = (b_i - a_i) / max(a_i, b_i)
    
    Interpretation:
    - s_i ≈ 1: Well clustered, far from other clusters
    - s_i ≈ 0: Near decision boundary
    - s_i < 0: May be in wrong cluster
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        
        # Compute a_i: avg distance to same cluster
        same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            same_cluster_points = X[same_cluster_mask]
            a_i = np.mean([np.sqrt(np.sum((X[i] - p)**2)) for p in same_cluster_points])
        else:
            a_i = 0  # Only one point in cluster
        
        # Compute b_i: min avg distance to other clusters
        b_i = np.inf
        for label in unique_labels:
            if label == current_label:
                continue
            
            other_cluster_mask = (labels == label)
            if np.sum(other_cluster_mask) > 0:
                other_cluster_points = X[other_cluster_mask]
                avg_dist = np.mean([np.sqrt(np.sum((X[i] - p)**2)) for p in other_cluster_points])
                if avg_dist < b_i:
                    b_i = avg_dist
        
        if b_i == np.inf:
            b_i = 0  # Only one cluster
        
        # Compute silhouette
        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0
    
    return silhouette_vals


def silhouette_score(X, labels):
    """
    Compute average silhouette score for clustering.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    labels : array_like, shape (n_samples,)
        Cluster assignments
    
    Returns:
    --------
    float
        Average silhouette score in range [-1, 1]
    
    Notes:
    ------
    - Higher is better
    - Maximizing silhouette helps find optimal K
    """
    return np.mean(silhouette_samples(X, labels))


def elbow_method(X, k_range=range(1, 11), random_state=None):
    """
    Compute WCSS for range of K values (elbow method).
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    k_range : iterable
        Range of K values to try
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'k_values': List of K values
        - 'wcss': WCSS for each K
        - 'optimal_k': Suggested K (knee point)
    
    Notes:
    ------
    The "elbow" is where WCSS decreases significantly
    slow down. Beyond this point, adding clusters
    provides marginal improvement.
    """
    from .kmeans import kmeans
    
    k_values = list(k_range)
    wcss_values = []
    
    for k in k_values:
        if k < 1:
            continue
        
        result = kmeans(X, k, random_state=random_state)
        wcss_values.append(result['wcss'])
    
    # Find elbow using rate of change
    optimal_k = find_elbow(k_values, wcss_values)
    
    return {
        'k_values': k_values,
        'wcss': wcss_values,
        'optimal_k': optimal_k
    }


def find_elbow(k_values, wcss_values):
    """
    Find elbow point in WCSS curve.
    
    Uses the "knee" detection algorithm based on
    perpendicular distance from line connecting endpoints.
    
    Parameters:
    -----------
    k_values : list
        K values
    wcss_values : list
        WCSS for each K
    
    Returns:
    --------
    int
        Optimal K value
    """
    if len(k_values) < 3:
        return k_values[0]
    
    # Normalize to [0, 1]
    k_norm = np.array(k_values, dtype=float)
    k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min() + 1e-10)
    
    wcss_norm = np.array(wcss_values, dtype=float)
    wcss_norm = (wcss_norm - wcss_norm.min()) / (wcss_norm.max() - wcss_norm.min() + 1e-10)
    
    # Line from first to last point
    p1 = np.array([k_norm[0], wcss_norm[0]])
    p2 = np.array([k_norm[-1], wcss_norm[-1]])
    
    # Find point with maximum perpendicular distance
    max_dist = 0
    elbow_idx = 0
    
    for i in range(1, len(k_values) - 1):
        p = np.array([k_norm[i], wcss_norm[i]])
        
        # Perpendicular distance to line
        d = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        
        if d > max_dist:
            max_dist = d
            elbow_idx = i
    
    return k_values[elbow_idx]


def find_optimal_k(X, k_range=range(2, 11), method='silhouette', random_state=None):
    """
    Find optimal K using specified method.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    k_range : iterable
        Range of K values to try
    method : str
        'silhouette' (maximize) or 'elbow' (knee detection)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'optimal_k': Best K value
        - 'k_values': K values tested
        - 'scores': Score for each K
        - 'method': Method used
    """
    from .kmeans import kmeans
    
    k_values = list(k_range)
    scores = []
    
    for k in k_values:
        result = kmeans(X, k, random_state=random_state)
        
        if method == 'silhouette':
            if k == 1:
                score = -1  # Silhouette undefined for k=1
            else:
                score = silhouette_score(X, result['labels'])
        else:  # elbow/wcss
            score = result['wcss']
        
        scores.append(score)
    
    if method == 'silhouette':
        optimal_idx = np.argmax(scores)
    else:
        optimal_idx = k_values.index(find_elbow(k_values, scores))
    
    return {
        'optimal_k': k_values[optimal_idx],
        'k_values': k_values,
        'scores': scores,
        'method': method
    }
