"""
Error metrics for regression evaluation.

This module provides common error metrics including MAE, MAPE, MSE, RMSE,
NSE (Nash-Sutcliffe Efficiency), and R².
"""

import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error (MAE).
    
    MAE = (1/n) Σ |y_i - ŷ_i|
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        MAE value
    
    Notes:
    ------
    - L1 norm of errors
    - Robust to outliers
    - Not differentiable at zero
    - Same units as target variable
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE).
    
    MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        MAPE value (percentage)
    
    Notes:
    ------
    - Normalized version of MAE
    - Undefined when y_true contains zeros
    - Asymmetric (penalizes over-prediction more)
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def mse(y_true, y_pred):
    """
    Mean Squared Error (MSE).
    
    MSE = (1/n) Σ (y_i - ŷ_i)²
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        MSE value
    
    Notes:
    ------
    - L2 norm squared of errors
    - Sensitive to outliers (quadratic penalty)
    - Differentiable everywhere
    - Units are squared
    """
    return np.mean((y_true - y_pred)**2)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).
    
    RMSE = √[(1/n) Σ (y_i - ŷ_i)²]
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        RMSE value
    
    Notes:
    ------
    - Standard deviation of prediction errors
    - Sensitive to outliers
    - Same units as target variable
    - RMSE ≥ MAE always
    """
    return np.sqrt(mse(y_true, y_pred))


def nse(y_true, y_pred):
    """
    Nash-Sutcliffe Efficiency (NSE).
    
    NSE = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        NSE value
    
    Notes:
    ------
    - Dimensionless metric
    - NSE = 1: perfect prediction
    - NSE = 0: as good as mean predictor
    - NSE < 0: worse than mean predictor
    - Sensitive to extreme values
    - Widely used in hydrology and environmental modeling
    """
    y_mean = np.mean(y_true)
    
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - y_mean)**2)
    
    if denominator == 0:
        return -np.inf
    
    return 1 - (numerator / denominator)


def r_squared(y_true, y_pred):
    """
    Coefficient of determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        R² value
    
    Notes:
    ------
    - Proportion of variance explained
    - R² = 1: perfect fit
    - R² = 0: as good as mean
    - R² < 0: worse than mean
    - Equivalent to NSE for regression
    """
    return nse(y_true, y_pred)


def adjusted_r_squared(y_true, y_pred, n_features):
    """
    Adjusted R² (accounts for number of features).
    
    R²_adj = 1 - (1 - R²)(n - 1)/(n - p - 1)
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    n_features : int
        Number of features (excluding intercept)
    
    Returns:
    --------
    float
        Adjusted R² value
    """
    n = len(y_true)
    r2 = r_squared(y_true, y_pred)
    
    if n <= n_features + 1:
        return -np.inf
    
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    return adj_r2


def max_error(y_true, y_pred):
    """
    Maximum absolute error.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        Maximum error
    """
    return np.max(np.abs(y_true - y_pred))


def median_absolute_error(y_true, y_pred):
    """
    Median Absolute Error.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        Median absolute error
    
    Notes:
    ------
    More robust to outliers than MAE
    """
    return np.median(np.abs(y_true - y_pred))


def compute_all_metrics(y_true, y_pred, n_features=None):
    """
    Compute all regression metrics.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    n_features : int, optional
        Number of features (for adjusted R²)
    
    Returns:
    --------
    dict
        Dictionary of all metrics
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'MAPE': mape(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'NSE': nse(y_true, y_pred),
        'R²': r_squared(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Median AE': median_absolute_error(y_true, y_pred)
    }
    
    if n_features is not None:
        metrics['Adjusted R²'] = adjusted_r_squared(y_true, y_pred, n_features)
    
    return metrics


def residuals(y_true, y_pred):
    """
    Compute residuals.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    array (n_samples,)
        Residuals (y_true - y_pred)
    """
    return y_true - y_pred


def standardized_residuals(y_true, y_pred):
    """
    Compute standardized residuals.
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    array (n_samples,)
        Standardized residuals
    """
    res = residuals(y_true, y_pred)
    std = np.std(res)
    
    if std == 0:
        return np.zeros_like(res)
    
    return res / std


def percent_bias(y_true, y_pred):
    """
    Compute percent bias (PBIAS).
    
    PBIAS = 100 * Σ(y_i - ŷ_i) / Σy_i
    
    Parameters:
    -----------
    y_true : array (n_samples,)
        True values
    y_pred : array (n_samples,)
        Predicted values
    
    Returns:
    --------
    float
        Percent bias
    
    Notes:
    ------
    - Measures average tendency to over/under-predict
    - PBIAS = 0: unbiased
    - PBIAS > 0: under-prediction
    - PBIAS < 0: over-prediction
    """
    sum_true = np.sum(y_true)
    
    if sum_true == 0:
        return np.inf
    
    return 100 * np.sum(y_true - y_pred) / sum_true
