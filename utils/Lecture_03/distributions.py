"""
Distribution functions for Probability & Statistics.

This module provides PDF and CDF functions for various probability distributions,
along with moment calculations.
"""

import numpy as np
from scipy import special


def uniform_pdf(x, a=0, b=1):
    """
    Uniform distribution PDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate PDF
    a : float
        Lower bound
    b : float
        Upper bound
    
    Returns:
    --------
    array-like
        PDF values
    """
    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= a) & (x <= b)
    pdf[mask] = 1.0 / (b - a)
    return pdf


def uniform_cdf(x, a=0, b=1):
    """
    Uniform distribution CDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate CDF
    a : float
        Lower bound
    b : float
        Upper bound
    
    Returns:
    --------
    array-like
        CDF values
    """
    x = np.asarray(x)
    cdf = np.zeros_like(x, dtype=float)
    
    # Below a: CDF = 0
    # Between a and b: CDF = (x - a) / (b - a)
    mask_middle = (x >= a) & (x <= b)
    cdf[mask_middle] = (x[mask_middle] - a) / (b - a)
    
    # Above b: CDF = 1
    mask_above = x > b
    cdf[mask_above] = 1.0
    
    return cdf


def normal_pdf(x, mu=0, sigma=1):
    """
    Normal (Gaussian) distribution PDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate PDF
    mu : float
        Mean
    sigma : float
        Standard deviation
    
    Returns:
    --------
    array-like
        PDF values
    """
    x = np.asarray(x)
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def normal_cdf(x, mu=0, sigma=1):
    """
    Normal (Gaussian) distribution CDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate CDF
    mu : float
        Mean
    sigma : float
        Standard deviation
    
    Returns:
    --------
    array-like
        CDF values
    """
    x = np.asarray(x)
    z = (x - mu) / sigma
    return 0.5 * (1 + special.erf(z / np.sqrt(2)))


def gumbel_pdf(x, mu=0, beta=1):
    """
    Gumbel distribution PDF (Type I extreme value).
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate PDF
    mu : float
        Location parameter
    beta : float
        Scale parameter
    
    Returns:
    --------
    array-like
        PDF values
    
    Formula:
    --------
    f(x) = (1/β) * exp(-(z + exp(-z)))
    where z = (x - μ) / β
    """
    x = np.asarray(x)
    z = (x - mu) / beta
    return (1.0 / beta) * np.exp(-(z + np.exp(-z)))


def gumbel_cdf(x, mu=0, beta=1):
    """
    Gumbel distribution CDF (Type I extreme value).
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate CDF
    mu : float
        Location parameter
    beta : float
        Scale parameter
    
    Returns:
    --------
    array-like
        CDF values
    
    Formula:
    --------
    F(x) = exp(-exp(-z))
    where z = (x - μ) / β
    """
    x = np.asarray(x)
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))


def gev_pdf(x, mu=0, sigma=1, xi=0):
    """
    Generalized Extreme Value (GEV) distribution PDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate PDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter
    xi : float
        Shape parameter
        - xi = 0: Gumbel (Type I)
        - xi > 0: Fréchet (Type II)
        - xi < 0: Weibull (Type III)
    
    Returns:
    --------
    array-like
        PDF values
    """
    x = np.asarray(x)
    z = (x - mu) / sigma
    
    if abs(xi) < 1e-10:  # Gumbel case (xi ≈ 0)
        return gumbel_pdf(x, mu, sigma)
    
    # General GEV case
    pdf = np.zeros_like(x, dtype=float)
    
    # Valid domain
    if xi > 0:
        mask = z > -1/xi
    else:
        mask = z < -1/xi
    
    t = 1 + xi * z[mask]
    pdf[mask] = (1.0 / sigma) * t ** (-(1/xi) - 1) * np.exp(-t ** (-1/xi))
    
    return pdf


def gev_cdf(x, mu=0, sigma=1, xi=0):
    """
    Generalized Extreme Value (GEV) distribution CDF.
    
    Parameters:
    -----------
    x : array-like
        Values at which to evaluate CDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter
    xi : float
        Shape parameter
    
    Returns:
    --------
    array-like
        CDF values
    """
    x = np.asarray(x)
    z = (x - mu) / sigma
    
    if abs(xi) < 1e-10:  # Gumbel case (xi ≈ 0)
        return gumbel_cdf(x, mu, sigma)
    
    # General GEV case
    cdf = np.zeros_like(x, dtype=float)
    
    # Valid domain
    if xi > 0:
        mask = z > -1/xi
        cdf[z <= -1/xi] = 0  # Below lower bound
    else:
        mask = z < -1/xi
        cdf[z >= -1/xi] = 1  # Above upper bound
    
    t = 1 + xi * z[mask]
    cdf[mask] = np.exp(-t ** (-1/xi))
    
    return cdf


def compute_moments(data, return_all=False):
    """
    Compute statistical moments of data.
    
    Parameters:
    -----------
    data : array-like
        Data samples
    return_all : bool
        If True, return dict with all moments
        If False, return (mean, variance, skewness, kurtosis)
    
    Returns:
    --------
    tuple or dict
        Statistical moments
    """
    data = np.asarray(data)
    
    # First moment: Mean
    mean = np.mean(data)
    
    # Second central moment: Variance
    variance = np.var(data, ddof=1)  # Sample variance
    std = np.sqrt(variance)
    
    # Third standardized moment: Skewness
    if std > 0:
        skewness = np.mean(((data - mean) / std) ** 3)
    else:
        skewness = 0
    
    # Fourth standardized moment: Kurtosis (excess kurtosis)
    if std > 0:
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3
    else:
        kurtosis = 0
    
    if return_all:
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75)
        }
    else:
        return mean, variance, skewness, kurtosis


def sample_from_distribution(dist_name, size=1000, **params):
    """
    Generate random samples from a distribution.
    
    Parameters:
    -----------
    dist_name : str
        Distribution name ('uniform', 'normal', 'gumbel')
    size : int
        Number of samples
    **params : dict
        Distribution parameters
    
    Returns:
    --------
    array
        Random samples
    """
    if dist_name == 'uniform':
        a = params.get('a', 0)
        b = params.get('b', 1)
        return np.random.uniform(a, b, size)
    
    elif dist_name == 'normal':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        return np.random.normal(mu, sigma, size)
    
    elif dist_name == 'gumbel':
        mu = params.get('mu', 0)
        beta = params.get('beta', 1)
        return np.random.gumbel(mu, beta, size)
    
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
