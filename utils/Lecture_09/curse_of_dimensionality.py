"""
Curse of Dimensionality Demonstrations

This module provides functions to demonstrate various aspects
of the curse of dimensionality.
"""

import numpy as np


def distance_concentration_demo(n_points=1000, dimensions=range(1, 51), random_state=None):
    """
    Demonstrate distance concentration in high dimensions.
    
    In high-dimensional spaces, distances between random points
    become increasingly similar (concentrate).
    
    Parameters:
    -----------
    n_points : int
        Number of random points
    dimensions : iterable
        Range of dimensions to test
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'dimensions': List of dimensions
        - 'min_distances': Minimum pairwise distance per dimension
        - 'max_distances': Maximum pairwise distance per dimension
        - 'mean_distances': Mean pairwise distance per dimension
        - 'ratio': min/max ratio (approaches 1 in high dims)
    
    Notes:
    ------
    As dimensionality increases, the ratio of minimum to maximum
    distance approaches 1, making distance-based methods unreliable.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dims = list(dimensions)
    min_dists = []
    max_dists = []
    mean_dists = []
    ratios = []
    
    for d in dims:
        # Generate random points in d dimensions
        X = np.random.randn(n_points, d)
        
        # Compute pairwise distances (sample for efficiency)
        n_sample = min(100, n_points)
        indices = np.random.choice(n_points, n_sample, replace=False)
        X_sample = X[indices]
        
        distances = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                dist = np.linalg.norm(X_sample[i] - X_sample[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        ratio = min_dist / max_dist if max_dist > 0 else 0
        
        min_dists.append(min_dist)
        max_dists.append(max_dist)
        mean_dists.append(mean_dist)
        ratios.append(ratio)
    
    return {
        'dimensions': dims,
        'min_distances': np.array(min_dists),
        'max_distances': np.array(max_dists),
        'mean_distances': np.array(mean_dists),
        'ratio': np.array(ratios)
    }


def volume_growth_demo(dimensions=range(1, 21)):
    """
    Demonstrate exponential volume growth in high dimensions.
    
    Volume of unit hypercube: 1^d = 1 (constant)
    Volume of unit hypersphere: V_d ∝ r^d
    
    But the ratio of hypersphere to hypercube volume
    decreases exponentially!
    
    Parameters:
    -----------
    dimensions : iterable
        Range of dimensions
    
    Returns:
    --------
    dict
        Contains:
        - 'dimensions': List of dimensions
        - 'sphere_volume': Volume of unit hypersphere
        - 'cube_volume': Volume of unit hypercube (always 1)
        - 'ratio': Sphere/Cube ratio
    
    Notes:
    ------
    In high dimensions, most of the hypercube's volume is in the corners,
    far from the center. Data becomes increasingly sparse.
    """
    from scipy.special import gamma
    
    dims = list(dimensions)
    sphere_volumes = []
    cube_volumes = []
    ratios = []
    
    for d in dims:
        # Unit hypersphere volume: V_d = π^(d/2) / Γ(d/2 + 1)
        sphere_vol = (np.pi ** (d / 2)) / gamma(d / 2 + 1)
        
        # Unit hypercube volume
        cube_vol = 1.0
        
        ratio = sphere_vol / cube_vol
        
        sphere_volumes.append(sphere_vol)
        cube_volumes.append(cube_vol)
        ratios.append(ratio)
    
    return {
        'dimensions': dims,
        'sphere_volume': np.array(sphere_volumes),
        'cube_volume': np.array(cube_volumes),
        'ratio': np.array(ratios)
    }


def sampling_requirement(dimensions=range(1, 11), samples_per_dim=10):
    """
    Demonstrate exponential growth in sampling requirements.
    
    For a k-level factorial design in d dimensions,
    you need k^d samples to cover the space.
    
    Parameters:
    -----------
    dimensions : iterable
        Range of dimensions
    samples_per_dim : int
        Number of samples per dimension (k)
    
    Returns:
    --------
    dict
        Contains:
        - 'dimensions': List of dimensions
        - 'samples_required': k^d samples needed
        - 'log_samples': log10 of samples (for visualization)
    
    Notes:
    ------
    This exponential growth makes data collection infeasible
    in high dimensions.
    """
    dims = list(dimensions)
    samples_required = []
    log_samples = []
    
    for d in dims:
        n_samples = samples_per_dim ** d
        samples_required.append(n_samples)
        log_samples.append(np.log10(n_samples))
    
    return {
        'dimensions': dims,
        'samples_required': np.array(samples_required),
        'log_samples': np.array(log_samples),
        'samples_per_dim': samples_per_dim
    }


def nearest_neighbor_degradation(n_samples=200, dimensions=range(2, 31), 
                                  n_neighbors=5, random_state=None):
    """
    Demonstrate k-NN degradation in high dimensions.
    
    As dimensionality increases, k-NN classification accuracy
    degrades because distances become meaningless.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    dimensions : iterable
        Range of dimensions
    n_neighbors : int
        Number of neighbors for k-NN
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'dimensions': List of dimensions
        - 'accuracy': k-NN accuracy per dimension
        - 'avg_neighbor_distance': Average distance to k-th neighbor
    
    Notes:
    ------
    Creates a simple 2-class problem and measures k-NN accuracy
    as dimensionality increases.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dims = list(dimensions)
    accuracies = []
    avg_distances = []
    
    for d in dims:
        # Generate 2-class data
        n_per_class = n_samples // 2
        
        # Class 0: centered at origin
        X0 = np.random.randn(n_per_class, d) * 0.5
        
        # Class 1: centered at [1, 1, ..., 1]
        X1 = np.random.randn(n_per_class, d) * 0.5 + 1.0
        
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        
        # Train/test split
        n_train = int(0.7 * n_samples)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # k-NN classification
        predictions = []
        neighbor_dists = []
        
        for x_test in X_test:
            # Compute distances to all training points
            distances = np.linalg.norm(X_train - x_test, axis=1)
            
            # Find k nearest neighbors
            nearest_indices = np.argsort(distances)[:n_neighbors]
            nearest_labels = y_train[nearest_indices]
            
            # Majority vote
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)
            
            # Average distance to k-th neighbor
            neighbor_dists.append(distances[nearest_indices[-1]])
        
        # Accuracy
        accuracy = np.mean(np.array(predictions) == y_test)
        avg_dist = np.mean(neighbor_dists)
        
        accuracies.append(accuracy)
        avg_distances.append(avg_dist)
    
    return {
        'dimensions': dims,
        'accuracy': np.array(accuracies),
        'avg_neighbor_distance': np.array(avg_distances),
        'n_neighbors': n_neighbors
    }


def data_sparsity_demo(n_samples=1000, dimensions=range(1, 21), random_state=None):
    """
    Demonstrate data sparsity in high dimensions.
    
    Measures the fraction of space occupied by data points.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    dimensions : iterable
        Range of dimensions
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'dimensions': List of dimensions
        - 'avg_nearest_distance': Average distance to nearest neighbor
        - 'volume_per_point': Estimated volume per point
    
    Notes:
    ------
    As dimensionality increases, points become increasingly
    isolated from each other.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dims = list(dimensions)
    avg_nn_dists = []
    volumes_per_point = []
    
    for d in dims:
        # Generate random points in unit hypercube
        X = np.random.rand(n_samples, d)
        
        # Compute nearest neighbor distances (sample for efficiency)
        n_sample = min(100, n_samples)
        indices = np.random.choice(n_samples, n_sample, replace=False)
        
        nn_dists = []
        for i in indices:
            distances = np.linalg.norm(X - X[i], axis=1)
            distances[i] = np.inf  # Exclude self
            nn_dist = np.min(distances)
            nn_dists.append(nn_dist)
        
        avg_nn_dist = np.mean(nn_dists)
        
        # Volume per point (unit hypercube volume / n_samples)
        volume_per_point = 1.0 / n_samples
        
        avg_nn_dists.append(avg_nn_dist)
        volumes_per_point.append(volume_per_point)
    
    return {
        'dimensions': dims,
        'avg_nearest_distance': np.array(avg_nn_dists),
        'volume_per_point': np.array(volumes_per_point)
    }
