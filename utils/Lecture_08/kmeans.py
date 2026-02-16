"""
K-Means Clustering Algorithm

This module provides K-Means clustering from scratch, including
centroid initialization, cluster assignment, and convergence tracking.
"""

import numpy as np


def kmeans_plusplus_init(X, k, random_state=None):
    """
    K-Means++ initialization for better centroid placement.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    k : int
        Number of clusters
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    array_like, shape (k, n_features)
        Initial centroids
    
    Notes:
    ------
    K-Means++ selects initial centroids that are spread out,
    reducing the chance of poor convergence.
    
    Algorithm:
    1. Choose first centroid randomly from data points
    2. For each subsequent centroid:
       - Compute distance from each point to nearest centroid
       - Select next centroid with probability proportional to distance²
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    # First centroid: random choice
    idx = np.random.randint(n_samples)
    centroids[0] = X[idx]
    
    # Remaining centroids
    for i in range(1, k):
        # Compute squared distances to nearest centroid
        distances = np.zeros(n_samples)
        for j in range(n_samples):
            min_dist = np.inf
            for c in range(i):
                dist = np.sum((X[j] - centroids[c])**2)
                if dist < min_dist:
                    min_dist = dist
            distances[j] = min_dist
        
        # Select next centroid with probability proportional to distance²
        probabilities = distances / np.sum(distances)
        idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[idx]
    
    return centroids


def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    centroids : array_like, shape (k, n_features)
        Cluster centroids
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Cluster labels (0 to k-1)
    """
    n_samples = X.shape[0]
    k = centroids.shape[0]
    
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        min_dist = np.inf
        for j in range(k):
            dist = np.sum((X[i] - centroids[j])**2)
            if dist < min_dist:
                min_dist = dist
                labels[i] = j
    
    return labels


def update_centroids(X, labels, k):
    """
    Update centroids as mean of assigned points.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    labels : array_like, shape (n_samples,)
        Cluster labels
    k : int
        Number of clusters
    
    Returns:
    --------
    array_like, shape (k, n_features)
        Updated centroids
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for j in range(k):
        mask = (labels == j)
        if np.sum(mask) > 0:
            centroids[j] = np.mean(X[mask], axis=0)
        else:
            # Empty cluster: reinitialize randomly
            centroids[j] = X[np.random.randint(X.shape[0])]
    
    return centroids


def compute_wcss(X, labels, centroids):
    """
    Compute Within-Cluster Sum of Squares (WCSS / Inertia).
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    labels : array_like, shape (n_samples,)
        Cluster labels
    centroids : array_like, shape (k, n_features)
        Cluster centroids
    
    Returns:
    --------
    float
        Total WCSS
    
    Notes:
    ------
    WCSS = Σ_i Σ_{x ∈ C_i} ||x - μ_i||²
    
    Lower WCSS indicates tighter clusters.
    """
    wcss = 0.0
    for i, x in enumerate(X):
        wcss += np.sum((x - centroids[labels[i]])**2)
    return wcss


def kmeans(X, k, max_iterations=100, tol=1e-4, init='kmeans++', random_state=None):
    """
    K-Means clustering algorithm.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    k : int
        Number of clusters
    max_iterations : int
        Maximum iterations
    tol : float
        Convergence tolerance (centroid movement)
    init : str
        Initialization method: 'kmeans++' or 'random'
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'labels': Cluster assignments
        - 'centroids': Final centroids
        - 'wcss': Final WCSS
        - 'n_iterations': Iterations until convergence
        - 'wcss_history': WCSS at each iteration
        - 'centroids_history': Centroids at each iteration
    
    Notes:
    ------
    Algorithm:
    1. Initialize k centroids
    2. Repeat until convergence:
       a. Assign each point to nearest centroid
       b. Update centroids as mean of assigned points
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    
    # Initialize centroids
    if init == 'kmeans++':
        centroids = kmeans_plusplus_init(X, k, random_state)
    else:
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[indices].copy()
    
    # Track history
    wcss_history = []
    centroids_history = [centroids.copy()]
    
    for iteration in range(max_iterations):
        # Assignment step
        labels = assign_clusters(X, centroids)
        
        # Update step
        new_centroids = update_centroids(X, labels, k)
        
        # Compute WCSS
        wcss = compute_wcss(X, labels, new_centroids)
        wcss_history.append(wcss)
        centroids_history.append(new_centroids.copy())
        
        # Check convergence
        centroid_shift = np.sum((new_centroids - centroids)**2)
        centroids = new_centroids
        
        if centroid_shift < tol:
            break
    
    return {
        'labels': labels,
        'centroids': centroids,
        'wcss': wcss,
        'n_iterations': iteration + 1,
        'wcss_history': wcss_history,
        'centroids_history': centroids_history
    }


def kmeans_step_by_step(X, k, random_state=None):
    """
    Generator that yields K-Means state at each step.
    
    Useful for animation and visualization.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Data points
    k : int
        Number of clusters
    random_state : int, optional
        Random seed
    
    Yields:
    -------
    dict
        State at each step:
        - 'step': Step number
        - 'phase': 'init', 'assign', or 'update'
        - 'centroids': Current centroids
        - 'labels': Current labels (None for init)
        - 'wcss': Current WCSS (None for init)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize
    centroids = kmeans_plusplus_init(X, k, random_state)
    
    yield {
        'step': 0,
        'phase': 'init',
        'centroids': centroids.copy(),
        'labels': None,
        'wcss': None
    }
    
    step = 1
    prev_centroids = None
    
    while True:
        # Assignment step
        labels = assign_clusters(X, centroids)
        wcss = compute_wcss(X, labels, centroids)
        
        yield {
            'step': step,
            'phase': 'assign',
            'centroids': centroids.copy(),
            'labels': labels.copy(),
            'wcss': wcss
        }
        
        # Update step
        new_centroids = update_centroids(X, labels, k)
        
        yield {
            'step': step,
            'phase': 'update',
            'centroids': new_centroids.copy(),
            'labels': labels.copy(),
            'wcss': wcss
        }
        
        # Check convergence
        if prev_centroids is not None:
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
        
        prev_centroids = centroids.copy()
        centroids = new_centroids
        step += 1
        
        if step > 50:  # Safety limit
            break
