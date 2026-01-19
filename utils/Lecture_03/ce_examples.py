"""
Civil Engineering examples for Probability & Statistics.

This module provides CE-specific data generation functions for
concrete strength, flood data, sensor noise, structural loads, and wind speeds.
"""

import numpy as np


def generate_concrete_strength_data(n_samples=100, mean_strength=30, std_strength=5, 
                                    add_outliers=False, seed=None):
    """
    Generate synthetic concrete compressive strength data.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    mean_strength : float
        Mean compressive strength (MPa)
    std_strength : float
        Standard deviation (MPa)
    add_outliers : bool
        Add some outlier samples
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Contains 'strengths', 'ages', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate normally distributed strengths
    strengths = np.random.normal(mean_strength, std_strength, n_samples)
    
    # Ensure no negative values
    strengths = np.maximum(strengths, 0)
    
    # Add outliers if requested
    if add_outliers:
        n_outliers = int(0.05 * n_samples)  # 5% outliers
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        # Some low outliers (poor quality)
        strengths[outlier_indices[:n_outliers//2]] *= 0.5
        # Some high outliers (exceptional quality)
        strengths[outlier_indices[n_outliers//2:]] *= 1.3
    
    # Generate corresponding ages (days)
    ages = np.random.randint(7, 90, n_samples)
    
    return {
        'strengths': strengths,
        'ages': ages,
        'info': {
            'mean': mean_strength,
            'std': std_strength,
            'n_samples': n_samples,
            'has_outliers': add_outliers
        }
    }


def generate_flood_data(n_years=50, location_param=100, scale_param=20, 
                       distribution='gumbel', seed=None):
    """
    Generate synthetic annual peak flood discharge data.
    
    Parameters:
    -----------
    n_years : int
        Number of years of data
    location_param : float
        Location parameter (m³/s)
    scale_param : float
        Scale parameter
    distribution : str
        'gumbel' or 'gev'
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'discharges', 'years', 'return_periods', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate extreme value data
    if distribution == 'gumbel':
        discharges = np.random.gumbel(location_param, scale_param, n_years)
    elif distribution == 'gev':
        # Using scipy for GEV
        from scipy.stats import genextreme
        xi = 0.1  # Shape parameter
        discharges = genextreme.rvs(xi, loc=location_param, scale=scale_param, size=n_years)
    else:
        raise ValueError("distribution must be 'gumbel' or 'gev'")
    
    # Ensure positive values
    discharges = np.maximum(discharges, 0)
    
    # Years
    years = np.arange(1, n_years + 1)
    
    # Calculate return periods for sorted data
    sorted_discharges = np.sort(discharges)[::-1]  # Descending
    ranks = np.arange(1, n_years + 1)
    return_periods = (n_years + 1) / ranks
    
    return {
        'discharges': discharges,
        'years': years,
        'sorted_discharges': sorted_discharges,
        'return_periods': return_periods,
        'info': {
            'n_years': n_years,
            'location': location_param,
            'scale': scale_param,
            'distribution': distribution
        }
    }


def generate_sensor_noise(n_readings=1000, true_value=25.0, noise_std=0.5, 
                         noise_type='gaussian', seed=None):
    """
    Generate synthetic sensor readings with noise.
    
    Parameters:
    -----------
    n_readings : int
        Number of sensor readings
    true_value : float
        True value being measured
    noise_std : float
        Standard deviation of noise
    noise_type : str
        'gaussian' or 'uniform'
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'readings', 'noise', 'true_value', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate noise
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std, n_readings)
    elif noise_type == 'uniform':
        # Uniform noise with same std
        a = -noise_std * np.sqrt(3)
        b = noise_std * np.sqrt(3)
        noise = np.random.uniform(a, b, n_readings)
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'")
    
    # Sensor readings = true value + noise
    readings = true_value + noise
    
    return {
        'readings': readings,
        'noise': noise,
        'true_value': true_value,
        'info': {
            'n_readings': n_readings,
            'noise_std': noise_std,
            'noise_type': noise_type,
            'measured_mean': np.mean(readings),
            'measured_std': np.std(readings)
        }
    }


def generate_load_data(n_samples=200, dead_load_mean=50, dead_load_std=5,
                      live_load_mean=30, live_load_std=10, seed=None):
    """
    Generate synthetic structural load data.
    
    Parameters:
    -----------
    n_samples : int
        Number of load samples
    dead_load_mean : float
        Mean dead load (kN)
    dead_load_std : float
        Std of dead load (kN)
    live_load_mean : float
        Mean live load (kN)
    live_load_std : float
        Std of live load (kN)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'dead_loads', 'live_loads', 'total_loads', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Dead loads (less variable)
    dead_loads = np.random.normal(dead_load_mean, dead_load_std, n_samples)
    dead_loads = np.maximum(dead_loads, 0)
    
    # Live loads (more variable)
    live_loads = np.random.normal(live_load_mean, live_load_std, n_samples)
    live_loads = np.maximum(live_loads, 0)
    
    # Total loads
    total_loads = dead_loads + live_loads
    
    # Calculate covariance
    cov_matrix = np.cov(dead_loads, live_loads)
    correlation = np.corrcoef(dead_loads, live_loads)[0, 1]
    
    return {
        'dead_loads': dead_loads,
        'live_loads': live_loads,
        'total_loads': total_loads,
        'info': {
            'n_samples': n_samples,
            'dead_load_mean': dead_load_mean,
            'live_load_mean': live_load_mean,
            'covariance_matrix': cov_matrix,
            'correlation': correlation
        }
    }


def generate_wind_speed_data(n_years=30, mean_speed=25, std_speed=5,
                            extreme_factor=1.5, seed=None):
    """
    Generate synthetic wind speed data including extreme events.
    
    Parameters:
    -----------
    n_years : int
        Number of years
    mean_speed : float
        Mean annual max wind speed (m/s)
    std_speed : float
        Standard deviation (m/s)
    extreme_factor : float
        Factor for extreme events
    seed : int, optional
        Random seed
    
    Returns:
    --------
    dict
        Contains 'annual_max_speeds', 'design_speeds', 'info'
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate annual maximum wind speeds using Gumbel distribution
    # Gumbel is appropriate for maxima
    location = mean_speed - 0.5772 * std_speed  # Euler-Mascheroni constant
    scale = std_speed / 1.2825
    
    annual_max_speeds = np.random.gumbel(location, scale, n_years)
    annual_max_speeds = np.maximum(annual_max_speeds, 0)
    
    # Calculate design wind speeds for different return periods
    return_periods = np.array([10, 25, 50, 100])
    design_speeds = {}
    
    for T in return_periods:
        # Gumbel quantile function
        p = 1 - 1/T
        y = -np.log(-np.log(p))
        design_speed = location + scale * y
        design_speeds[f'{T}-year'] = design_speed
    
    return {
        'annual_max_speeds': annual_max_speeds,
        'design_speeds': design_speeds,
        'return_periods': return_periods,
        'info': {
            'n_years': n_years,
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'location': location,
            'scale': scale
        }
    }


def generate_correlated_data(n_samples=100, mean1=0, mean2=0, std1=1, std2=1,
                            correlation=0.7, seed=None):
    """
    Generate two correlated random variables.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    mean1, mean2 : float
        Means of the two variables
    std1, std2 : float
        Standard deviations
    correlation : float
        Desired correlation coefficient (-1 to 1)
    seed : int, optional
        Random seed
    
    Returns:
    --------
    tuple
        (x, y) correlated data arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uncorrelated standard normal variables
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    # Create correlation
    x = z1
    y = correlation * z1 + np.sqrt(1 - correlation**2) * z2
    
    # Scale and shift
    x = mean1 + std1 * x
    y = mean2 + std2 * y
    
    return x, y
