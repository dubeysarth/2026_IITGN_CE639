"""
Polynomial regression implementations.

This module provides functions for polynomial feature transformation
and polynomial regression fitting.
"""

import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree=2, include_bias=True):
    """
    Generate polynomial features up to specified degree.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Input features
    degree : int
        Maximum polynomial degree
    include_bias : bool
        Whether to include bias term (column of ones)
    
    Returns:
    --------
    X_poly : array (n_samples, n_poly_features)
        Polynomial features
    
    Example:
    --------
    For X = [[a, b]] and degree=2:
    Returns [[1, a, b, a^2, ab, b^2]]
    """
    n_samples, n_features = X.shape
    
    # Generate all polynomial combinations
    features = []
    
    if include_bias:
        features.append(np.ones(n_samples))
    
    # For each degree from 1 to degree
    for d in range(1, degree + 1):
        # Generate all combinations with replacement
        for combo in combinations_with_replacement(range(n_features), d):
            # Compute product of features in combination
            feature = np.ones(n_samples)
            for idx in combo:
                feature *= X[:, idx]
            features.append(feature)
    
    X_poly = np.column_stack(features)
    
    return X_poly


def polynomial_features_simple(X, degree=2):
    """
    Generate polynomial features for 1D input (simpler version).
    
    Parameters:
    -----------
    X : array (n_samples,) or (n_samples, 1)
        Input feature
    degree : int
        Maximum polynomial degree
    
    Returns:
    --------
    X_poly : array (n_samples, degree+1)
        Polynomial features [1, x, x^2, ..., x^degree]
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples = len(X)
    X_poly = np.zeros((n_samples, degree + 1))
    
    for d in range(degree + 1):
        X_poly[:, d] = (X[:, 0] ** d)
    
    return X_poly


def polynomial_fit(X, y, degree=2, lambda_reg=0.0):
    """
    Fit polynomial regression.
    
    Parameters:
    -----------
    X : array (n_samples,) or (n_samples, n_features)
        Input features
    y : array (n_samples,)
        Target values
    degree : int
        Polynomial degree
    lambda_reg : float
        Ridge regularization parameter (0 for no regularization)
    
    Returns:
    --------
    dict
        Contains 'weights', 'degree', 'lambda_reg'
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Generate polynomial features
    X_poly = polynomial_features(X, degree=degree, include_bias=True)
    
    # Fit using ridge regression (or OLS if lambda_reg=0)
    n_features = X_poly.shape[1]
    
    if lambda_reg > 0:
        # Ridge regression
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't penalize intercept
        
        XtX = X_poly.T @ X_poly
        Xty = X_poly.T @ y
        
        w = np.linalg.solve(XtX + lambda_reg * I, Xty)
    else:
        # OLS
        w = np.linalg.lstsq(X_poly, y, rcond=None)[0]
    
    return {
        'weights': w,
        'degree': degree,
        'lambda_reg': lambda_reg
    }


def polynomial_predict(X, model):
    """
    Make predictions using polynomial model.
    
    Parameters:
    -----------
    X : array (n_samples,) or (n_samples, n_features)
        Input features
    model : dict
        Model from polynomial_fit
    
    Returns:
    --------
    y_pred : array (n_samples,)
        Predictions
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Generate polynomial features
    X_poly = polynomial_features(X, degree=model['degree'], include_bias=True)
    
    # Predict
    y_pred = X_poly @ model['weights']
    
    return y_pred


def fit_polynomial_range(X, y, max_degree=10, lambda_reg=0.0):
    """
    Fit polynomial models for a range of degrees.
    
    Parameters:
    -----------
    X : array (n_samples,) or (n_samples, n_features)
        Input features
    y : array (n_samples,)
        Target values
    max_degree : int
        Maximum degree to try
    lambda_reg : float
        Regularization parameter
    
    Returns:
    --------
    list
        List of models for each degree
    """
    models = []
    
    for degree in range(1, max_degree + 1):
        model = polynomial_fit(X, y, degree=degree, lambda_reg=lambda_reg)
        models.append(model)
    
    return models


def polynomial_basis_functions(x, degree):
    """
    Evaluate polynomial basis functions at point x.
    
    Parameters:
    -----------
    x : float or array
        Point(s) to evaluate
    degree : int
        Polynomial degree
    
    Returns:
    --------
    array
        Basis function values [1, x, x^2, ..., x^degree]
    """
    x = np.asarray(x)
    basis = np.zeros((*x.shape, degree + 1))
    
    for d in range(degree + 1):
        basis[..., d] = x ** d
    
    return basis


def vandermonde_matrix(x, degree):
    """
    Construct Vandermonde matrix for polynomial fitting.
    
    Parameters:
    -----------
    x : array (n_samples,)
        Input points
    degree : int
        Polynomial degree
    
    Returns:
    --------
    V : array (n_samples, degree+1)
        Vandermonde matrix
    
    Notes:
    ------
    V[i, j] = x[i]^j
    """
    n = len(x)
    V = np.zeros((n, degree + 1))
    
    for j in range(degree + 1):
        V[:, j] = x ** j
    
    return V


def polynomial_derivative(weights, x):
    """
    Compute derivative of polynomial at point x.
    
    Parameters:
    -----------
    weights : array (degree+1,)
        Polynomial coefficients [w0, w1, w2, ...]
        Represents w0 + w1*x + w2*x^2 + ...
    x : float or array
        Point(s) to evaluate derivative
    
    Returns:
    --------
    float or array
        Derivative value(s)
    """
    degree = len(weights) - 1
    x = np.asarray(x)
    
    # Derivative coefficients
    deriv_weights = np.array([i * weights[i] for i in range(1, degree + 1)])
    
    # Evaluate derivative polynomial
    result = np.zeros_like(x, dtype=float)
    for i, w in enumerate(deriv_weights):
        result += w * (x ** i)
    
    return result


def polynomial_integral(weights, a, b):
    """
    Compute definite integral of polynomial from a to b.
    
    Parameters:
    -----------
    weights : array (degree+1,)
        Polynomial coefficients
    a : float
        Lower bound
    b : float
        Upper bound
    
    Returns:
    --------
    float
        Integral value
    """
    degree = len(weights) - 1
    
    # Antiderivative coefficients
    antideriv_weights = np.zeros(degree + 2)
    for i in range(degree + 1):
        antideriv_weights[i + 1] = weights[i] / (i + 1)
    
    # Evaluate at bounds
    F_b = sum(w * (b ** i) for i, w in enumerate(antideriv_weights))
    F_a = sum(w * (a ** i) for i, w in enumerate(antideriv_weights))
    
    return F_b - F_a
