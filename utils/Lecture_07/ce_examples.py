"""
Civil Engineering Classification Datasets

This module provides synthetic CE datasets for classification problems
including crack detection, structural failure, soil classification,
flood prediction, and pavement condition assessment.
"""

import numpy as np


def crack_detection_data(n_samples=500, imbalance_ratio=0.3, noise_std=0.3, 
                        random_state=None):
    """
    Generate synthetic crack detection dataset.
    
    Binary classification: crack (1) vs no crack (0)
    
    Features:
    - Image intensity variance
    - Edge density
    - Texture uniformity
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    imbalance_ratio : float
        Ratio of crack samples (minority class)
    noise_std : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_crack = int(n_samples * imbalance_ratio)
    n_no_crack = n_samples - n_crack
    
    # No crack: low variance, low edge density, high uniformity
    X_no_crack = np.column_stack([
        np.random.uniform(0, 0.3, n_no_crack),      # Low intensity variance
        np.random.uniform(0, 0.2, n_no_crack),      # Low edge density
        np.random.uniform(0.7, 1.0, n_no_crack)     # High uniformity
    ])
    y_no_crack = np.zeros(n_no_crack, dtype=int)
    
    # Crack: high variance, high edge density, low uniformity
    X_crack = np.column_stack([
        np.random.uniform(0.6, 1.0, n_crack),       # High intensity variance
        np.random.uniform(0.7, 1.0, n_crack),       # High edge density
        np.random.uniform(0, 0.4, n_crack)          # Low uniformity
    ])
    y_crack = np.ones(n_crack, dtype=int)
    
    # Combine and add noise
    X = np.vstack([X_no_crack, X_crack])
    X += np.random.normal(0, noise_std, X.shape)
    X = np.clip(X, 0, 1)  # Keep in [0, 1]
    
    y = np.concatenate([y_no_crack, y_crack])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'feature_names': ['Intensity Variance', 'Edge Density', 'Texture Uniformity'],
        'target_name': 'Crack Detected',
        'description': f'Crack detection dataset: {n_samples} samples, '
                      f'{imbalance_ratio*100:.0f}% crack samples'
    }


def structural_failure_data(n_samples=1000, failure_rate=0.05, noise_std=0.1,
                           random_state=None):
    """
    Generate synthetic structural failure prediction dataset.
    
    Binary classification: failure (1) vs safe (0)
    Highly imbalanced (rare event prediction)
    
    Features:
    - Load ratio (applied load / design load)
    - Material degradation (0-1)
    - Age (years)
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    failure_rate : float
        Ratio of failure samples (rare event)
    noise_std : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_failure = int(n_samples * failure_rate)
    n_safe = n_samples - n_failure
    
    # Safe structures: low load ratio, low degradation, younger
    X_safe = np.column_stack([
        np.random.uniform(0.3, 0.7, n_safe),        # Moderate load ratio
        np.random.uniform(0, 0.3, n_safe),          # Low degradation
        np.random.uniform(0, 20, n_safe)            # Younger age
    ])
    y_safe = np.zeros(n_safe, dtype=int)
    
    # Failed structures: high load ratio, high degradation, older
    X_failure = np.column_stack([
        np.random.uniform(0.8, 1.5, n_failure),     # High load ratio
        np.random.uniform(0.6, 1.0, n_failure),     # High degradation
        np.random.uniform(25, 50, n_failure)        # Older age
    ])
    y_failure = np.ones(n_failure, dtype=int)
    
    # Combine and add noise
    X = np.vstack([X_safe, X_failure])
    X[:, 0] += np.random.normal(0, noise_std, X.shape[0])  # Load ratio noise
    X[:, 1] += np.random.normal(0, noise_std * 0.5, X.shape[0])  # Degradation noise
    X[:, 2] += np.random.normal(0, noise_std * 10, X.shape[0])  # Age noise
    
    # Clip to valid ranges
    X[:, 0] = np.clip(X[:, 0], 0, 2)
    X[:, 1] = np.clip(X[:, 1], 0, 1)
    X[:, 2] = np.clip(X[:, 2], 0, 100)
    
    y = np.concatenate([y_safe, y_failure])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'feature_names': ['Load Ratio', 'Material Degradation', 'Age (years)'],
        'target_name': 'Structural Failure',
        'description': f'Structural failure dataset: {n_samples} samples, '
                      f'{failure_rate*100:.1f}% failure rate (rare event)'
    }


def soil_classification_data(n_samples=600, noise_std=0.2, random_state=None):
    """
    Generate synthetic soil classification dataset.
    
    Multi-class classification: Clay (0), Sand (1), Gravel (2)
    
    Features:
    - Particle size (mm)
    - Plasticity index
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    noise_std : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'class_names', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_per_class = n_samples // 3
    
    # Clay: small particles, high plasticity
    X_clay = np.column_stack([
        np.random.uniform(0, 0.002, n_per_class),   # Very small particles
        np.random.uniform(15, 40, n_per_class)      # High plasticity
    ])
    y_clay = np.zeros(n_per_class, dtype=int)
    
    # Sand: medium particles, low plasticity
    X_sand = np.column_stack([
        np.random.uniform(0.05, 2.0, n_per_class),  # Medium particles
        np.random.uniform(0, 5, n_per_class)        # Low plasticity
    ])
    y_sand = np.ones(n_per_class, dtype=int)
    
    # Gravel: large particles, zero plasticity
    X_gravel = np.column_stack([
        np.random.uniform(2.0, 75, n_per_class),    # Large particles
        np.random.uniform(0, 1, n_per_class)        # No plasticity
    ])
    y_gravel = np.full(n_per_class, 2, dtype=int)
    
    # Combine and add noise
    X = np.vstack([X_clay, X_sand, X_gravel])
    X[:, 0] += np.random.normal(0, noise_std * 0.1, X.shape[0])
    X[:, 1] += np.random.normal(0, noise_std * 2, X.shape[0])
    
    # Clip to valid ranges
    X[:, 0] = np.clip(X[:, 0], 0, 100)
    X[:, 1] = np.clip(X[:, 1], 0, 50)
    
    y = np.concatenate([y_clay, y_sand, y_gravel])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'feature_names': ['Particle Size (mm)', 'Plasticity Index'],
        'target_name': 'Soil Type',
        'class_names': ['Clay', 'Sand', 'Gravel'],
        'description': f'Soil classification dataset: {len(y)} samples, 3 classes'
    }


def flood_prediction_data(n_samples=800, flood_rate=0.15, noise_std=0.3,
                         random_state=None):
    """
    Generate synthetic flood prediction dataset.
    
    Binary classification: flood (1) vs no flood (0)
    Imbalanced dataset (floods are rare)
    
    Features:
    - Rainfall (mm/day)
    - River level (m above normal)
    - Soil saturation (0-1)
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    flood_rate : float
        Ratio of flood events
    noise_std : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_flood = int(n_samples * flood_rate)
    n_no_flood = n_samples - n_flood
    
    # No flood: low rainfall, low river level, low saturation
    X_no_flood = np.column_stack([
        np.random.uniform(0, 30, n_no_flood),       # Low rainfall
        np.random.uniform(-1, 1, n_no_flood),       # Normal river level
        np.random.uniform(0, 0.5, n_no_flood)       # Low saturation
    ])
    y_no_flood = np.zeros(n_no_flood, dtype=int)
    
    # Flood: high rainfall, high river level, high saturation
    X_flood = np.column_stack([
        np.random.uniform(50, 150, n_flood),        # High rainfall
        np.random.uniform(2, 5, n_flood),           # High river level
        np.random.uniform(0.7, 1.0, n_flood)        # High saturation
    ])
    y_flood = np.ones(n_flood, dtype=int)
    
    # Combine and add noise
    X = np.vstack([X_no_flood, X_flood])
    X += np.random.normal(0, noise_std * np.array([5, 0.3, 0.05]), X.shape)
    
    # Clip to valid ranges
    X[:, 0] = np.clip(X[:, 0], 0, 200)
    X[:, 1] = np.clip(X[:, 1], -2, 10)
    X[:, 2] = np.clip(X[:, 2], 0, 1)
    
    y = np.concatenate([y_no_flood, y_flood])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'feature_names': ['Rainfall (mm/day)', 'River Level (m)', 'Soil Saturation'],
        'target_name': 'Flood Event',
        'description': f'Flood prediction dataset: {n_samples} samples, '
                      f'{flood_rate*100:.0f}% flood events'
    }


def pavement_condition_data(n_samples=900, noise_std=0.2, random_state=None):
    """
    Generate synthetic pavement condition assessment dataset.
    
    Multi-class classification: Good (0), Fair (1), Poor (2)
    
    Features:
    - Surface roughness (IRI, m/km)
    - Crack density (%)
    - Age (years)
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    noise_std : float
        Standard deviation of noise
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'class_names', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_per_class = n_samples // 3
    
    # Good condition: low roughness, low cracks, newer
    X_good = np.column_stack([
        np.random.uniform(1, 3, n_per_class),       # Low IRI
        np.random.uniform(0, 5, n_per_class),       # Few cracks
        np.random.uniform(0, 5, n_per_class)        # New pavement
    ])
    y_good = np.zeros(n_per_class, dtype=int)
    
    # Fair condition: moderate roughness, moderate cracks, medium age
    X_fair = np.column_stack([
        np.random.uniform(3, 6, n_per_class),       # Moderate IRI
        np.random.uniform(5, 15, n_per_class),      # Some cracks
        np.random.uniform(5, 15, n_per_class)       # Medium age
    ])
    y_fair = np.ones(n_per_class, dtype=int)
    
    # Poor condition: high roughness, many cracks, older
    X_poor = np.column_stack([
        np.random.uniform(6, 12, n_per_class),      # High IRI
        np.random.uniform(15, 40, n_per_class),     # Many cracks
        np.random.uniform(15, 30, n_per_class)      # Old pavement
    ])
    y_poor = np.full(n_per_class, 2, dtype=int)
    
    # Combine and add noise
    X = np.vstack([X_good, X_fair, X_poor])
    X += np.random.normal(0, noise_std * np.array([0.5, 2, 2]), X.shape)
    
    # Clip to valid ranges
    X[:, 0] = np.clip(X[:, 0], 0, 15)
    X[:, 1] = np.clip(X[:, 1], 0, 50)
    X[:, 2] = np.clip(X[:, 2], 0, 40)
    
    y = np.concatenate([y_good, y_fair, y_poor])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'feature_names': ['Surface Roughness (IRI)', 'Crack Density (%)', 'Age (years)'],
        'target_name': 'Pavement Condition',
        'class_names': ['Good', 'Fair', 'Poor'],
        'description': f'Pavement condition dataset: {len(y)} samples, 3 classes'
    }


def generate_linearly_separable_data(n_samples=200, n_features=2, 
                                    class_sep=2.0, random_state=None):
    """
    Generate linearly separable binary classification dataset.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_features : int
        Number of features
    class_sep : float
        Separation between classes (higher = easier)
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'description'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Class 0
    X_0 = np.random.randn(n_per_class, n_features)
    y_0 = np.zeros(n_per_class, dtype=int)
    
    # Class 1 (shifted)
    X_1 = np.random.randn(n_per_class, n_features) + class_sep
    y_1 = np.ones(n_per_class, dtype=int)
    
    # Combine
    X = np.vstack([X_0, X_1])
    y = np.concatenate([y_0, y_1])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        'X': X,
        'y': y,
        'description': f'Linearly separable dataset: {n_samples} samples, '
                      f'{n_features} features, separation={class_sep}'
    }
