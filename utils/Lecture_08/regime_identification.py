"""
Regime Identification

This module provides tools for detecting regimes in time series
using clustering-based approaches.
"""

import numpy as np


def extract_window_features(time_series, window_size=10, features='all'):
    """
    Extract features from sliding windows of time series.
    
    Parameters:
    -----------
    time_series : array_like, shape (n_timesteps,) or (n_timesteps, n_vars)
        Time series data
    window_size : int
        Size of sliding window
    features : str or list
        Features to extract: 'mean', 'std', 'min', 'max', 'range', 'all'
    
    Returns:
    --------
    dict
        Contains:
        - 'features': Feature matrix, shape (n_windows, n_features)
        - 'window_indices': Start index for each window
        - 'feature_names': Names of extracted features
    
    Notes:
    ------
    Each window becomes a feature vector for clustering.
    """
    ts = np.atleast_2d(time_series)
    if ts.shape[0] < ts.shape[1]:
        ts = ts.T  # Ensure (n_timesteps, n_vars)
    
    n_timesteps, n_vars = ts.shape
    n_windows = n_timesteps - window_size + 1
    
    if n_windows <= 0:
        raise ValueError(f"Window size {window_size} too large for series length {n_timesteps}")
    
    # Feature extraction
    if features == 'all':
        feature_list = ['mean', 'std', 'min', 'max', 'range']
    else:
        feature_list = [features] if isinstance(features, str) else features
    
    feature_matrix = []
    feature_names = []
    
    for start_idx in range(n_windows):
        window = ts[start_idx:start_idx + window_size]
        window_features = []
        
        for var_idx in range(n_vars):
            var_window = window[:, var_idx]
            var_name = f'var{var_idx}_' if n_vars > 1 else ''
            
            if start_idx == 0:  # Build feature names once
                for feat in feature_list:
                    feature_names.append(f'{var_name}{feat}')
            
            for feat in feature_list:
                if feat == 'mean':
                    window_features.append(np.mean(var_window))
                elif feat == 'std':
                    window_features.append(np.std(var_window))
                elif feat == 'min':
                    window_features.append(np.min(var_window))
                elif feat == 'max':
                    window_features.append(np.max(var_window))
                elif feat == 'range':
                    window_features.append(np.max(var_window) - np.min(var_window))
        
        feature_matrix.append(window_features)
    
    return {
        'features': np.array(feature_matrix),
        'window_indices': np.arange(n_windows),
        'feature_names': feature_names
    }


def detect_change_points(time_series, threshold=2.0, method='diff'):
    """
    Simple change point detection in time series.
    
    Parameters:
    -----------
    time_series : array_like
        1D time series
    threshold : float
        Threshold for detecting changes (in std units)
    method : str
        'diff': First difference magnitude
        'zscore': Z-score of values
    
    Returns:
    --------
    dict
        Contains:
        - 'change_points': Indices of detected changes
        - 'scores': Change scores at each point
    """
    ts = np.asarray(time_series).flatten()
    n = len(ts)
    
    if method == 'diff':
        # Absolute first differences
        diffs = np.abs(np.diff(ts))
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs) + 1e-10
        scores = (diffs - mean_diff) / std_diff
        
        # Insert 0 at start to align with original indices
        scores = np.concatenate([[0], scores])
    else:  # zscore
        mean_ts = np.mean(ts)
        std_ts = np.std(ts) + 1e-10
        scores = np.abs((ts - mean_ts) / std_ts)
    
    change_points = np.where(np.abs(scores) > threshold)[0]
    
    return {
        'change_points': change_points,
        'scores': scores
    }


def cluster_based_regime_detection(time_series, n_regimes=None, window_size=10, 
                                   random_state=None):
    """
    Detect regimes in time series using clustering.
    
    Parameters:
    -----------
    time_series : array_like
        Time series data
    n_regimes : int, optional
        Number of regimes. If None, uses elbow method.
    window_size : int
        Window size for feature extraction
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'regime_labels': Regime label for each window
        - 'regime_centers': Centroid of each regime (in feature space)
        - 'n_regimes': Number of detected regimes
        - 'features': Extracted window features
    
    Notes:
    ------
    Workflow:
    1. Extract features from sliding windows
    2. Apply K-Means clustering
    3. Each cluster represents a "regime"
    """
    from .kmeans import kmeans
    from .cluster_validation import find_optimal_k
    
    # Extract window features
    result = extract_window_features(time_series, window_size)
    X = result['features']
    
    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - X_mean) / X_std
    
    # Determine number of regimes
    if n_regimes is None:
        optimal = find_optimal_k(X_scaled, k_range=range(2, 6), 
                                 method='silhouette', random_state=random_state)
        n_regimes = optimal['optimal_k']
    
    # Cluster
    km_result = kmeans(X_scaled, n_regimes, random_state=random_state)
    
    return {
        'regime_labels': km_result['labels'],
        'regime_centers': km_result['centroids'],
        'n_regimes': n_regimes,
        'features': X,
        'window_indices': result['window_indices'],
        'feature_names': result['feature_names']
    }


def map_regimes_to_time(regime_labels, window_indices, n_timesteps, mode='start'):
    """
    Map window-based regime labels back to time series indices.
    
    Parameters:
    -----------
    regime_labels : array_like
        Regime label for each window
    window_indices : array_like
        Start index for each window
    n_timesteps : int
        Total length of original time series
    mode : str
        'start': Use regime at window start
        'majority': Use most common regime in overlapping windows
    
    Returns:
    --------
    array_like, shape (n_timesteps,)
        Regime label for each time step
    """
    if mode == 'start':
        time_regimes = np.full(n_timesteps, -1)
        for i, (win_idx, label) in enumerate(zip(window_indices, regime_labels)):
            if time_regimes[win_idx] == -1:
                time_regimes[win_idx] = label
        
        # Forward fill
        last_regime = regime_labels[0]
        for i in range(n_timesteps):
            if time_regimes[i] == -1:
                time_regimes[i] = last_regime
            else:
                last_regime = time_regimes[i]
        
        return time_regimes
    else:
        raise NotImplementedError("Only 'start' mode implemented")


def plot_regime_timeline(time, values, regime_labels, title='Regime Timeline',
                        figsize=(14, 6)):
    """
    Plot time series with regime coloring.
    
    Parameters:
    -----------
    time : array_like
        Time values
    values : array_like
        Time series values
    regime_labels : array_like
        Regime label for each time point
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    colors = plt.cm.Set2(np.linspace(0, 1, n_regimes))
    
    # Plot time series with regime coloring
    for i, regime in enumerate(unique_regimes):
        mask = (regime_labels == regime)
        ax.scatter(time[mask], values[mask], c=[colors[i]], 
                  s=20, alpha=0.7, label=f'Regime {regime}')
    
    # Plot line
    ax.plot(time, values, 'k-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return fig, ax
