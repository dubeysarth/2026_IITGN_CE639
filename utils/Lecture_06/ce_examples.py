"""
Civil Engineering example datasets for regression.

This module provides synthetic CE datasets for concrete strength,
traffic flow, and pavement deflection modeling.
"""

import numpy as np


def concrete_strength_data(n_samples=100, noise_std=3.0, seed=None):
    """
    Generate synthetic concrete strength dataset.
    
    Predicts compressive strength based on:
    - Cement content (kg/m³)
    - Water-cement ratio
    - Age (days)
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    cement = np.random.uniform(200, 400, n_samples)  # kg/m³
    water_cement_ratio = np.random.uniform(0.3, 0.6, n_samples)
    age = np.random.uniform(7, 90, n_samples)  # days
    
    # True relationship (simplified model)
    # Strength increases with cement, decreases with w/c ratio, increases with age
    strength = (
        0.15 * cement 
        - 80 * water_cement_ratio 
        + 0.3 * age 
        - 20
    )
    
    # Add noise
    strength += np.random.normal(0, noise_std, n_samples)
    
    # Ensure positive strength
    strength = np.maximum(strength, 10)
    
    X = np.column_stack([cement, water_cement_ratio, age])
    
    return {
        'X': X,
        'y': strength,
        'feature_names': ['Cement (kg/m³)', 'W/C Ratio', 'Age (days)'],
        'target_name': 'Compressive Strength (MPa)',
        'description': 'Concrete compressive strength prediction based on mix design and age'
    }


def traffic_flow_data(n_samples=100, noise_std=50, seed=None):
    """
    Generate synthetic traffic flow dataset.
    
    Predicts traffic flow (vehicles/hour) based on:
    - Time of day (hour)
    - Day of week (0=Monday, 6=Sunday)
    - Weather condition (0=clear, 1=rain)
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    hour = np.random.uniform(0, 24, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    weather = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # True relationship
    # Peak hours: 8-9 AM and 5-6 PM
    # Lower flow on weekends
    # Reduced flow in rain
    
    # Base flow with morning and evening peaks
    flow = 500 + 300 * np.exp(-((hour - 8)**2) / 8) + 400 * np.exp(-((hour - 17)**2) / 8)
    
    # Weekend reduction
    weekend_mask = (day_of_week >= 5)
    flow[weekend_mask] *= 0.7
    
    # Weather reduction
    flow[weather == 1] *= 0.8
    
    # Add noise
    flow += np.random.normal(0, noise_std, n_samples)
    
    # Ensure positive flow
    flow = np.maximum(flow, 50)
    
    X = np.column_stack([hour, day_of_week, weather])
    
    return {
        'X': X,
        'y': flow,
        'feature_names': ['Hour', 'Day of Week', 'Weather (0=clear, 1=rain)'],
        'target_name': 'Traffic Flow (vehicles/hour)',
        'description': 'Traffic flow prediction based on time and weather conditions'
    }


def pavement_deflection_data(n_samples=100, noise_std=0.05, seed=None):
    """
    Generate synthetic pavement deflection dataset.
    
    Predicts surface deflection (mm) based on:
    - Load (kN)
    - Pavement thickness (mm)
    - Temperature (°C)
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    load = np.random.uniform(20, 100, n_samples)  # kN
    thickness = np.random.uniform(150, 300, n_samples)  # mm
    temperature = np.random.uniform(10, 40, n_samples)  # °C
    
    # True relationship
    # Deflection increases with load, decreases with thickness
    # Temperature affects stiffness (higher temp = more deflection)
    
    deflection = (
        0.02 * load 
        - 0.003 * thickness 
        + 0.01 * temperature 
        + 0.5
    )
    
    # Add noise
    deflection += np.random.normal(0, noise_std, n_samples)
    
    # Ensure positive deflection
    deflection = np.maximum(deflection, 0.1)
    
    X = np.column_stack([load, thickness, temperature])
    
    return {
        'X': X,
        'y': deflection,
        'feature_names': ['Load (kN)', 'Thickness (mm)', 'Temperature (°C)'],
        'target_name': 'Surface Deflection (mm)',
        'description': 'Pavement deflection prediction based on load and structural properties'
    }


def beam_deflection_data(n_samples=100, noise_std=0.5, seed=None):
    """
    Generate synthetic beam deflection dataset.
    
    Predicts maximum deflection based on:
    - Span length (m)
    - Load (kN/m)
    - Moment of inertia (m⁴)
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    span = np.random.uniform(3, 10, n_samples)  # m
    load = np.random.uniform(5, 20, n_samples)  # kN/m
    I = np.random.uniform(0.0001, 0.001, n_samples)  # m⁴
    
    # True relationship (simplified beam theory)
    # δ = 5wL⁴/(384EI) for uniformly distributed load
    E = 200e9  # Pa (steel modulus)
    
    deflection = (5 * load * 1000 * span**4) / (384 * E * I) * 1000  # mm
    
    # Add noise
    deflection += np.random.normal(0, noise_std, n_samples)
    
    # Ensure positive deflection
    deflection = np.maximum(deflection, 0.1)
    
    X = np.column_stack([span, load, I])
    
    return {
        'X': X,
        'y': deflection,
        'feature_names': ['Span (m)', 'Load (kN/m)', 'Moment of Inertia (m⁴)'],
        'target_name': 'Max Deflection (mm)',
        'description': 'Beam deflection prediction based on structural properties'
    }


def soil_settlement_data(n_samples=100, noise_std=5.0, seed=None):
    """
    Generate synthetic soil settlement dataset.
    
    Predicts settlement (mm) based on:
    - Applied pressure (kPa)
    - Soil thickness (m)
    - Void ratio
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'feature_names', 'target_name', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features
    pressure = np.random.uniform(50, 300, n_samples)  # kPa
    thickness = np.random.uniform(2, 10, n_samples)  # m
    void_ratio = np.random.uniform(0.5, 1.2, n_samples)
    
    # True relationship (simplified consolidation)
    # Settlement increases with pressure, thickness, and void ratio
    
    settlement = (
        0.3 * pressure 
        + 15 * thickness 
        + 50 * void_ratio 
        - 50
    )
    
    # Add noise
    settlement += np.random.normal(0, noise_std, n_samples)
    
    # Ensure positive settlement
    settlement = np.maximum(settlement, 5)
    
    X = np.column_stack([pressure, thickness, void_ratio])
    
    return {
        'X': X,
        'y': settlement,
        'feature_names': ['Pressure (kPa)', 'Thickness (m)', 'Void Ratio'],
        'target_name': 'Settlement (mm)',
        'description': 'Soil settlement prediction based on loading and soil properties'
    }


def generate_nonlinear_data(n_samples=100, noise_std=0.5, seed=None):
    """
    Generate nonlinear 1D dataset for demonstrating polynomial/kernel regression.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    noise_std : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'X', 'y', 'description'
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.random.uniform(-3, 3, n_samples)
    y = np.sin(X) + 0.3 * X**2 - 0.1 * X**3
    y += np.random.normal(0, noise_std, n_samples)
    
    return {
        'X': X,
        'y': y,
        'description': 'Nonlinear function: y = sin(x) + 0.3x² - 0.1x³'
    }
