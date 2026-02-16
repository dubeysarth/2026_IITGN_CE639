"""
Class Imbalance Handling

This module provides techniques for handling imbalanced datasets including
SMOTE, random over/undersampling, and weighted loss functions.
"""

import numpy as np


def random_oversample(X, y, target_ratio=1.0, random_state=None):
    """
    Random oversampling of minority class.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Features
    y : array_like, shape (n_samples,)
        Binary labels
    target_ratio : float
        Desired ratio of minority/majority (1.0 = balanced)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    X_resampled, y_resampled : arrays
        Resampled dataset
    
    Notes:
    ------
    Duplicates random samples from minority class until target ratio achieved.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify majority and minority classes
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    minority_class = unique_classes[np.argmin(counts)]
    
    n_majority = np.max(counts)
    n_minority = np.min(counts)
    
    # Calculate target minority count
    n_minority_target = int(n_majority * target_ratio)
    n_to_add = n_minority_target - n_minority
    
    if n_to_add <= 0:
        return X.copy(), y.copy()
    
    # Get minority samples
    minority_indices = np.where(y == minority_class)[0]
    
    # Randomly sample with replacement
    oversample_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)
    
    # Combine
    X_resampled = np.vstack([X, X[oversample_indices]])
    y_resampled = np.concatenate([y, y[oversample_indices]])
    
    return X_resampled, y_resampled


def random_undersample(X, y, target_ratio=1.0, random_state=None):
    """
    Random undersampling of majority class.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Features
    y : array_like, shape (n_samples,)
        Binary labels
    target_ratio : float
        Desired ratio of minority/majority (1.0 = balanced)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    X_resampled, y_resampled : arrays
        Resampled dataset
    
    Notes:
    ------
    Randomly removes samples from majority class until target ratio achieved.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify majority and minority classes
    unique_classes, counts = np.unique(y, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]
    minority_class = unique_classes[np.argmin(counts)]
    
    n_minority = np.min(counts)
    
    # Calculate target majority count
    n_majority_target = int(n_minority / target_ratio)
    
    # Get indices
    majority_indices = np.where(y == majority_class)[0]
    minority_indices = np.where(y == minority_class)[0]
    
    # Randomly sample majority class
    undersample_indices = np.random.choice(majority_indices, size=n_majority_target, 
                                          replace=False)
    
    # Combine
    keep_indices = np.concatenate([minority_indices, undersample_indices])
    X_resampled = X[keep_indices]
    y_resampled = y[keep_indices]
    
    return X_resampled, y_resampled


def smote(X, y, k_neighbors=5, target_ratio=1.0, random_state=None):
    """
    SMOTE: Synthetic Minority Oversampling Technique.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Features
    y : array_like, shape (n_samples,)
        Binary labels
    k_neighbors : int
        Number of nearest neighbors to use
    target_ratio : float
        Desired ratio of minority/majority (1.0 = balanced)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    X_resampled, y_resampled : arrays
        Resampled dataset with synthetic samples
    
    Notes:
    ------
    Creates synthetic samples by interpolating between minority samples
    and their k-nearest neighbors.
    
    Algorithm:
    1. For each minority sample x_i
    2. Find k nearest minority neighbors
    3. Randomly select one neighbor x_nn
    4. Create synthetic sample: x_new = x_i + λ(x_nn - x_i), λ ~ U(0,1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Identify minority class
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    
    n_majority = np.max(counts)
    n_minority = np.min(counts)
    
    # Calculate number of synthetic samples needed
    n_minority_target = int(n_majority * target_ratio)
    n_synthetic = n_minority_target - n_minority
    
    if n_synthetic <= 0:
        return X.copy(), y.copy()
    
    # Get minority samples
    minority_indices = np.where(y == minority_class)[0]
    X_minority = X[minority_indices]
    
    # Generate synthetic samples
    synthetic_samples = []
    
    for _ in range(n_synthetic):
        # Randomly select a minority sample
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        
        # Find k nearest neighbors (excluding itself)
        distances = np.linalg.norm(X_minority - sample, axis=1)
        nearest_indices = np.argsort(distances)[1:k_neighbors+1]
        
        # Randomly select one neighbor
        nn_idx = np.random.choice(nearest_indices)
        neighbor = X_minority[nn_idx]
        
        # Create synthetic sample
        lambda_val = np.random.uniform(0, 1)
        synthetic = sample + lambda_val * (neighbor - sample)
        synthetic_samples.append(synthetic)
    
    # Combine with original data
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(n_synthetic, minority_class)
    
    X_resampled = np.vstack([X, X_synthetic])
    y_resampled = np.concatenate([y, y_synthetic])
    
    return X_resampled, y_resampled


def weighted_cross_entropy(y_true, y_pred, class_weights=None, epsilon=1e-15):
    """
    Weighted binary cross-entropy loss.
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array_like, shape (n_samples,)
        Predicted probabilities [0, 1]
    class_weights : dict or array_like, optional
        Weights for each class. If dict: {0: w0, 1: w1}
        If array: [w0, w1]
        If None: equal weights
    epsilon : float
        Small constant to prevent log(0)
    
    Returns:
    --------
    float
        Weighted average cross-entropy loss
    
    Notes:
    ------
    L = -w_y (y log(ŷ) + (1-y) log(1-ŷ))
    
    Common weighting schemes:
    - Inverse frequency: w_i = n_total / (n_classes * n_i)
    - Balanced: w_i = n_total / (2 * n_i)
    """
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Parse class weights
    if class_weights is None:
        weights = np.ones_like(y_true)
    elif isinstance(class_weights, dict):
        weights = np.array([class_weights[int(label)] for label in y_true])
    else:
        weights = np.array([class_weights[int(label)] for label in y_true])
    
    # Compute weighted loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    weighted_loss = weights * loss
    
    return np.mean(weighted_loss)


def compute_class_weights(y, method='balanced'):
    """
    Compute class weights for imbalanced datasets.
    
    Parameters:
    -----------
    y : array_like
        Class labels
    method : str
        Weighting method:
        - 'balanced': n_total / (n_classes * n_i)
        - 'inverse': 1 / n_i (normalized)
    
    Returns:
    --------
    dict
        Class weights {class_label: weight}
    
    Examples:
    ---------
    >>> y = np.array([0, 0, 0, 0, 1])  # 80% class 0, 20% class 1
    >>> compute_class_weights(y, method='balanced')
    {0: 0.625, 1: 2.5}
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)
    
    if method == 'balanced':
        weights = {cls: n_samples / (n_classes * count) 
                  for cls, count in zip(unique_classes, counts)}
    elif method == 'inverse':
        inv_counts = 1.0 / counts
        normalized = inv_counts / np.sum(inv_counts) * n_classes
        weights = {cls: w for cls, w in zip(unique_classes, normalized)}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights


def generate_imbalanced_data(n_samples=1000, imbalance_ratio=0.1, 
                            n_features=2, class_sep=1.0, random_state=None):
    """
    Generate synthetic imbalanced binary classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    imbalance_ratio : float
        Ratio of minority class (e.g., 0.1 = 10% minority)
    n_features : int
        Number of features
    class_sep : float
        Separation between classes (higher = easier)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Features, shape (n_samples, n_features)
        - 'y': Labels, shape (n_samples,)
        - 'description': Dataset description
    
    Notes:
    ------
    Generates two Gaussian clusters with different means.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Calculate class sizes
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority
    
    # Generate majority class (class 0)
    X_majority = np.random.randn(n_majority, n_features)
    y_majority = np.zeros(n_majority, dtype=int)
    
    # Generate minority class (class 1) with shifted mean
    X_minority = np.random.randn(n_minority, n_features) + class_sep
    y_minority = np.ones(n_minority, dtype=int)
    
    # Combine
    X = np.vstack([X_majority, X_minority])
    y = np.concatenate([y_majority, y_minority])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    description = (f"Imbalanced dataset: {n_samples} samples, "
                  f"{imbalance_ratio*100:.1f}% minority class, "
                  f"{n_features} features")
    
    return {
        'X': X,
        'y': y,
        'description': description
    }


def plot_class_distribution(y, title='Class Distribution', figsize=(8, 5)):
    """
    Plot class distribution as bar chart.
    
    Parameters:
    -----------
    y : array_like
        Class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    unique_classes, counts = np.unique(y, return_counts=True)
    percentages = 100 * counts / len(y)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(unique_classes, counts, color=['#2E86AB', '#A23B72'], 
                  edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Annotate bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(unique_classes)
    ax.grid(axis='y', alpha=0.3)
    
    return fig, ax
