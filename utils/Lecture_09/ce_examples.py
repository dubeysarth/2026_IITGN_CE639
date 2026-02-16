"""
Civil Engineering Dimensionality Reduction Examples

This module provides synthetic datasets for CE dimensionality reduction applications:
- Bridge sensor data (multi-sensor SHM)
- Material spectral analysis
- Traffic multivariate data
- Structural mode shapes
"""

import numpy as np


def bridge_sensor_data(n_timesteps=500, n_sensors=20, n_modes=3, random_state=None):
    """
    Generate synthetic bridge sensor data with underlying modes.
    
    Simulates SHM sensors measuring vibrations with multiple
    structural modes contributing to the signal.
    
    Parameters:
    -----------
    n_timesteps : int
        Number of time steps
    n_sensors : int
        Number of sensors
    n_modes : int
        Number of underlying structural modes
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Sensor data, shape (n_timesteps, n_sensors)
        - 'true_modes': True mode shapes
        - 'mode_contributions': Contribution of each mode over time
        - 'sensor_locations': Normalized sensor positions
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Reduce high-dimensional sensor data to identify
    dominant structural modes for damage detection.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sensor locations along bridge span
    sensor_locations = np.linspace(0, 1, n_sensors)
    
    # True mode shapes (sine waves with different frequencies)
    true_modes = np.zeros((n_modes, n_sensors))
    for i in range(n_modes):
        true_modes[i] = np.sin((i + 1) * np.pi * sensor_locations)
    
    # Mode contributions over time (damped oscillations)
    time = np.linspace(0, 10, n_timesteps)
    mode_contributions = np.zeros((n_timesteps, n_modes))
    
    for i in range(n_modes):
        freq = (i + 1) * 2.0  # Hz
        damping = 0.1 * (i + 1)
        amplitude = 1.0 / (i + 1)
        
        mode_contributions[:, i] = amplitude * np.exp(-damping * time) * np.sin(2 * np.pi * freq * time)
    
    # Sensor data = mode_contributions @ true_modes + noise
    X = mode_contributions @ true_modes
    X += np.random.randn(n_timesteps, n_sensors) * 0.1  # Measurement noise
    
    return {
        'X': X,
        'true_modes': true_modes,
        'mode_contributions': mode_contributions,
        'sensor_locations': sensor_locations,
        'time': time,
        'description': 'Bridge SHM sensor data with structural modes'
    }


def material_spectral_data(n_samples=150, n_wavelengths=100, n_materials=3, random_state=None):
    """
    Generate synthetic material spectral analysis data.
    
    Simulates spectroscopy measurements (e.g., infrared, Raman)
    for material identification.
    
    Parameters:
    -----------
    n_samples : int
        Number of material samples
    n_wavelengths : int
        Number of wavelength measurements
    n_materials : int
        Number of material types
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Spectral data, shape (n_samples, n_wavelengths)
        - 'y': Material labels
        - 'wavelengths': Wavelength values
        - 'material_names': Names of materials
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Reduce high-dimensional spectral data for
    material classification and quality control.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    wavelengths = np.linspace(400, 2500, n_wavelengths)  # nm
    
    material_names = ['Concrete', 'Steel', 'Asphalt'][:n_materials]
    
    # Define spectral signatures (peaks at different wavelengths)
    material_signatures = {
        'Concrete': [(800, 50, 0.8), (1200, 30, 0.6), (1800, 40, 0.5)],
        'Steel': [(600, 40, 0.9), (1000, 60, 0.7), (1600, 50, 0.6)],
        'Asphalt': [(700, 35, 0.7), (1100, 45, 0.8), (1500, 55, 0.65)]
    }
    
    X = []
    y = []
    
    n_per_material = n_samples // n_materials
    
    for mat_idx, material in enumerate(material_names):
        signatures = material_signatures[material]
        
        for _ in range(n_per_material):
            # Base spectrum
            spectrum = np.zeros(n_wavelengths)
            
            # Add peaks
            for peak_center, peak_width, peak_height in signatures:
                peak = peak_height * np.exp(-((wavelengths - peak_center) ** 2) / (2 * peak_width ** 2))
                spectrum += peak
            
            # Add noise and baseline
            spectrum += np.random.randn(n_wavelengths) * 0.05
            spectrum += 0.1  # Baseline
            
            X.append(spectrum)
            y.append(mat_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return {
        'X': X,
        'y': y,
        'wavelengths': wavelengths,
        'material_names': material_names,
        'description': 'Material spectral analysis for classification'
    }


def traffic_multivariate_data(n_timesteps=1000, random_state=None):
    """
    Generate synthetic multivariate traffic data.
    
    Includes flow, speed, occupancy, weather, time-of-day features.
    
    Parameters:
    -----------
    n_timesteps : int
        Number of time steps
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Feature matrix, shape (n_timesteps, n_features)
        - 'feature_names': Names of features
        - 'time': Time array
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: Reduce high-dimensional traffic data for
    congestion prediction and pattern analysis.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    time = np.arange(n_timesteps)
    
    # Time-of-day features (24-hour cycle)
    hour_of_day = (time % 24) / 24
    sin_hour = np.sin(2 * np.pi * hour_of_day)
    cos_hour = np.cos(2 * np.pi * hour_of_day)
    
    # Day-of-week features (7-day cycle)
    day_of_week = (time // 24) % 7
    is_weekend = (day_of_week >= 5).astype(float)
    
    # Traffic patterns (correlated with time)
    base_flow = 1000 + 500 * sin_hour + 300 * (1 - is_weekend)
    flow = base_flow + np.random.randn(n_timesteps) * 100
    flow = np.clip(flow, 0, 2000)
    
    # Speed (inversely related to flow)
    speed = 80 - 0.03 * flow + np.random.randn(n_timesteps) * 5
    speed = np.clip(speed, 10, 100)
    
    # Occupancy (related to flow and speed)
    occupancy = flow / speed * 0.5 + np.random.randn(n_timesteps) * 2
    occupancy = np.clip(occupancy, 0, 100)
    
    # Weather features
    temperature = 20 + 10 * np.sin(2 * np.pi * time / 365) + np.random.randn(n_timesteps) * 3
    precipitation = np.maximum(0, np.random.randn(n_timesteps) * 5)
    
    # Incident indicator (rare events)
    incidents = (np.random.rand(n_timesteps) < 0.05).astype(float)
    
    # Lagged features
    flow_lag1 = np.roll(flow, 1)
    flow_lag1[0] = flow[0]
    speed_lag1 = np.roll(speed, 1)
    speed_lag1[0] = speed[0]
    
    # Combine features
    X = np.column_stack([
        flow, speed, occupancy,
        sin_hour, cos_hour, is_weekend,
        temperature, precipitation, incidents,
        flow_lag1, speed_lag1
    ])
    
    feature_names = [
        'flow', 'speed', 'occupancy',
        'sin_hour', 'cos_hour', 'is_weekend',
        'temperature', 'precipitation', 'incidents',
        'flow_lag1', 'speed_lag1'
    ]
    
    return {
        'X': X,
        'feature_names': feature_names,
        'time': time,
        'description': 'Multivariate traffic data for dimensionality reduction'
    }


def structural_mode_shapes(n_nodes=50, n_modes=5, random_state=None):
    """
    Generate synthetic structural mode shapes.
    
    Simulates vibration mode shapes for a beam or bridge.
    
    Parameters:
    -----------
    n_nodes : int
        Number of nodes along structure
    n_modes : int
        Number of vibration modes
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'mode_shapes': Mode shape matrix, shape (n_modes, n_nodes)
        - 'frequencies': Natural frequencies (Hz)
        - 'node_locations': Normalized node positions
        - 'description': Dataset description
    
    Notes:
    ------
    CE Application: PCA can identify dominant mode shapes from
    vibration measurements for structural health monitoring.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    node_locations = np.linspace(0, 1, n_nodes)
    
    mode_shapes = np.zeros((n_modes, n_nodes))
    frequencies = np.zeros(n_modes)
    
    for i in range(n_modes):
        # Mode shape: sin((i+1) * π * x)
        mode_shapes[i] = np.sin((i + 1) * np.pi * node_locations)
        
        # Natural frequency (proportional to mode number squared)
        frequencies[i] = (i + 1) ** 2 * 2.0  # Hz
        
        # Add small perturbations
        mode_shapes[i] += np.random.randn(n_nodes) * 0.05
    
    # Normalize mode shapes
    for i in range(n_modes):
        mode_shapes[i] /= np.linalg.norm(mode_shapes[i])
    
    return {
        'mode_shapes': mode_shapes,
        'frequencies': frequencies,
        'node_locations': node_locations,
        'description': 'Structural vibration mode shapes'
    }


def high_dimensional_ce_data(n_samples=200, n_features=50, n_informative=5, random_state=None):
    """
    Generate high-dimensional CE data with few informative features.
    
    Simulates a scenario where many measurements are taken but only
    a few contain useful information.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features
    random_state : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains:
        - 'X': Feature matrix
        - 'y': Target variable (continuous)
        - 'informative_indices': Indices of informative features
        - 'description': Dataset description
    
    Notes:
    ------
    Demonstrates curse of dimensionality and benefits of
    dimensionality reduction.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Informative features
    informative_indices = np.random.choice(n_features, n_informative, replace=False)
    
    X = np.zeros((n_samples, n_features))
    
    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)
    X[:, informative_indices] = X_informative
    
    # Generate noise features
    noise_indices = np.setdiff1d(np.arange(n_features), informative_indices)
    X[:, noise_indices] = np.random.randn(n_samples, len(noise_indices)) * 0.5
    
    # Target: linear combination of informative features + noise
    weights = np.random.randn(n_informative)
    y = X_informative @ weights + np.random.randn(n_samples) * 0.1
    
    return {
        'X': X,
        'y': y,
        'informative_indices': informative_indices,
        'description': 'High-dimensional data with few informative features'
    }
