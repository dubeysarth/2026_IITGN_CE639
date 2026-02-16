"""
Civil Engineering Clustering Examples

This module provides synthetic datasets for CE clustering applications:
- Sensor clustering (SHM, IoT)
- Traffic flow regimes
- Material property clustering
- Vibration regime identification
"""

import numpy as np


def generate_blob_data(n_samples=300, centers=None, n_clusters=3, 
                       cluster_std=1.0, random_state=None):
    """
    Generate synthetic blob data for clustering.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    centers : array_like, optional
        Custom cluster centers
    n_clusters : int
        Number of clusters (if centers not provided)
    cluster_std : float or list
        Standard deviation per cluster
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Data matrix, shape (n_samples, 2)
        - 'y_true': True cluster labels
        - 'centers': Cluster centers
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if centers is None:
        centers = np.random.uniform(-5, 5, (n_clusters, 2))
    else:
        centers = np.array(centers)
        n_clusters = len(centers)
    
    if isinstance(cluster_std, (int, float)):
        cluster_std = [cluster_std] * n_clusters
    
    n_per_cluster = n_samples // n_clusters
    
    X = []
    y = []
    
    for i, (center, std) in enumerate(zip(centers, cluster_std)):
        n = n_per_cluster if i < n_clusters - 1 else n_samples - len(y)
        cluster_data = np.random.normal(loc=center, scale=std, size=(n, 2))
        X.append(cluster_data)
        y.extend([i] * n)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return {'X': X, 'y_true': y, 'centers': centers}


def sensor_clustering_data(n_sensors=50, n_clusters=3, random_state=None):
    """
    Generate synthetic sensor data for SHM clustering.
    
    Sensors are characterized by:
    - Average reading magnitude
    - Reading variability (std)
    - Response time (feature)
    
    Parameters:
    -----------
    n_sensors : int
        Number of sensors
    n_clusters : int
        Number of sensor groups
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Feature matrix (avg_reading, variability, response_time)
        - 'y_true': True cluster labels
        - 'feature_names': Names of features
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Grouping sensors in a bridge monitoring system
    to identify similar behavior patterns or anomalies.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Define cluster characteristics
    # Cluster 0: Normal sensors (low reading, low variability)
    # Cluster 1: Active zones (medium reading, high variability)
    # Cluster 2: Stressed zones (high reading, medium variability)
    
    cluster_params = [
        {'mean': [10, 2, 5], 'std': [2, 0.5, 1]},   # Normal
        {'mean': [25, 8, 10], 'std': [5, 2, 2]},    # Active
        {'mean': [50, 5, 3], 'std': [10, 1.5, 0.5]} # Stressed
    ]
    
    n_per_cluster = n_sensors // n_clusters
    
    X = []
    y = []
    
    for i in range(n_clusters):
        n = n_per_cluster if i < n_clusters - 1 else n_sensors - len(y)
        params = cluster_params[i % len(cluster_params)]
        
        cluster_data = np.column_stack([
            np.random.normal(params['mean'][0], params['std'][0], n),
            np.random.normal(params['mean'][1], params['std'][1], n),
            np.random.normal(params['mean'][2], params['std'][2], n)
        ])
        
        X.append(cluster_data)
        y.extend([i] * n)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Ensure positive values
    X = np.abs(X)
    
    # Shuffle
    indices = np.random.permutation(n_sensors)
    X = X[indices]
    y = y[indices]
    
    return {
        'X': X,
        'y_true': y,
        'feature_names': ['avg_reading', 'variability', 'response_time'],
        'description': 'SHM sensor clustering: Group sensors by behavior patterns'
    }


def traffic_flow_data(n_timesteps=500, n_regimes=3, random_state=None):
    """
    Generate synthetic traffic flow time series with regimes.
    
    Regimes:
    - Free flow: High speed, low density
    - Synchronized: Medium speed, medium density
    - Congested: Low speed, high density
    
    Parameters:
    -----------
    n_timesteps : int
        Number of time steps
    n_regimes : int
        Number of traffic regimes
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'time': Time array
        - 'speed': Speed time series (km/h)
        - 'density': Density time series (veh/km)
        - 'flow': Flow time series (veh/h)
        - 'true_regimes': True regime labels
        - 'regime_names': Names of regimes
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Identifying traffic regimes for congestion
    prediction and traffic management.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Define regime characteristics
    regime_params = {
        0: {'speed': (80, 10), 'density': (20, 5)},    # Free flow
        1: {'speed': (50, 8), 'density': (60, 10)},     # Synchronized
        2: {'speed': (15, 5), 'density': (120, 15)},    # Congested
    }
    
    regime_names = ['Free Flow', 'Synchronized', 'Congested']
    
    # Generate regime sequence with transitions
    regime_lengths = np.random.randint(50, 150, size=n_timesteps // 50)
    regimes = []
    current_regime = 0
    
    for length in regime_lengths:
        regimes.extend([current_regime] * length)
        # Transition to next regime (circular)
        current_regime = (current_regime + np.random.choice([0, 1, 2], 
                         p=[0.3, 0.5, 0.2])) % n_regimes
        if len(regimes) >= n_timesteps:
            break
    
    regimes = np.array(regimes[:n_timesteps])
    
    # Generate speed and density based on regimes
    speed = np.zeros(n_timesteps)
    density = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        regime = regimes[t]
        params = regime_params[regime]
        speed[t] = np.random.normal(params['speed'][0], params['speed'][1])
        density[t] = np.random.normal(params['density'][0], params['density'][1])
    
    # Ensure positive values
    speed = np.clip(speed, 5, 120)
    density = np.clip(density, 5, 200)
    
    # Calculate flow (veh/h)
    flow = speed * density * 0.1  # Simplified relationship
    
    return {
        'time': np.arange(n_timesteps),
        'speed': speed,
        'density': density,
        'flow': flow,
        'true_regimes': regimes,
        'regime_names': regime_names,
        'description': 'Traffic flow regime identification'
    }


def material_property_data(n_samples=120, n_materials=4, random_state=None):
    """
    Generate synthetic material property data for clustering.
    
    Features:
    - Compressive strength (MPa)
    - Tensile strength (MPa)
    - Elastic modulus (GPa)
    - Density (kg/m³)
    
    Materials:
    - Concrete
    - Steel
    - Aluminum
    - Wood
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_materials : int
        Number of material types
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Feature matrix
        - 'y_true': True material labels
        - 'feature_names': Property names
        - 'material_names': Material names
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Automatic material classification based on
    measured properties (quality control, material selection).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Material properties: (mean, std)
    # [compressive, tensile, modulus, density]
    material_params = {
        'Concrete': [(35, 8), (3, 0.8), (30, 5), (2400, 100)],
        'Steel': [(400, 50), (400, 50), (200, 20), (7850, 50)],
        'Aluminum': [(100, 20), (150, 30), (70, 10), (2700, 50)],
        'Wood': [(40, 10), (80, 20), (12, 3), (600, 100)]
    }
    
    material_names = list(material_params.keys())[:n_materials]
    n_per_material = n_samples // n_materials
    
    X = []
    y = []
    
    for i, material in enumerate(material_names):
        n = n_per_material if i < n_materials - 1 else n_samples - len(y)
        params = material_params[material]
        
        material_data = np.column_stack([
            np.random.normal(params[0][0], params[0][1], n),  # Compressive
            np.random.normal(params[1][0], params[1][1], n),  # Tensile
            np.random.normal(params[2][0], params[2][1], n),  # Modulus
            np.random.normal(params[3][0], params[3][1], n)   # Density
        ])
        
        X.append(material_data)
        y.extend([i] * n)
    
    X = np.vstack(X)
    y = np.array(y)
    X = np.abs(X)  # Ensure positive
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return {
        'X': X,
        'y_true': y,
        'feature_names': ['compressive_strength_MPa', 'tensile_strength_MPa', 
                          'elastic_modulus_GPa', 'density_kg_m3'],
        'material_names': material_names,
        'description': 'Material property clustering for classification'
    }


def shm_vibration_data(n_timesteps=600, n_regimes=3, random_state=None):
    """
    Generate synthetic SHM vibration data with structural regimes.
    
    Regimes:
    - Normal: Low amplitude, stable frequency
    - Transitional: Moderate amplitude, variable frequency
    - Damaged/Alert: High amplitude, lower frequency
    
    Parameters:
    -----------
    n_timesteps : int
        Number of time steps
    n_regimes : int
        Number of structural states
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'time': Time array
        - 'acceleration': Acceleration time series
        - 'true_regimes': Regime labels
        - 'regime_names': Regime names
        - 'window_features': Pre-extracted window features
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Detecting structural health changes from
    vibration sensors on bridges, buildings, or wind turbines.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Regime parameters: (amplitude, frequency, noise)
    regime_params = {
        0: (0.5, 5.0, 0.1),   # Normal
        1: (1.5, 3.0, 0.3),   # Transitional
        2: (3.0, 1.5, 0.5)    # Damaged/Alert
    }
    
    regime_names = ['Normal', 'Transitional', 'Damaged/Alert']
    
    # Generate regime sequence
    segment_lengths = np.random.randint(100, 200, size=n_timesteps // 100)
    regimes = []
    current_regime = 0
    
    for length in segment_lengths:
        regimes.extend([current_regime] * length)
        # Progressive deterioration more likely
        prob_stay = 0.4
        prob_next = 0.5
        prob_prev = 0.1
        
        if current_regime == 0:
            current_regime = np.random.choice([0, 1], p=[prob_stay + prob_prev, prob_next])
        elif current_regime == n_regimes - 1:
            current_regime = np.random.choice([n_regimes-2, n_regimes-1], 
                                              p=[prob_prev, prob_stay + prob_next])
        else:
            current_regime = np.random.choice([current_regime-1, current_regime, current_regime+1],
                                              p=[prob_prev, prob_stay, prob_next])
        
        if len(regimes) >= n_timesteps:
            break
    
    regimes = np.array(regimes[:n_timesteps])
    
    # Generate vibration signal
    time = np.arange(n_timesteps) * 0.01  # 100 Hz sampling
    acceleration = np.zeros(n_timesteps)
    
    for t in range(n_timesteps):
        regime = regimes[t]
        amp, freq, noise = regime_params[regime]
        acceleration[t] = amp * np.sin(2 * np.pi * freq * time[t]) + np.random.normal(0, noise)
    
    # Extract simple window features
    window_size = 20
    n_windows = n_timesteps - window_size + 1
    window_features = np.zeros((n_windows, 3))  # mean, std, max
    
    for i in range(n_windows):
        window = acceleration[i:i + window_size]
        window_features[i] = [np.mean(np.abs(window)), np.std(window), np.max(np.abs(window))]
    
    return {
        'time': time,
        'acceleration': acceleration,
        'true_regimes': regimes,
        'regime_names': regime_names,
        'window_features': window_features,
        'description': 'SHM vibration regime identification for damage detection'
    }
