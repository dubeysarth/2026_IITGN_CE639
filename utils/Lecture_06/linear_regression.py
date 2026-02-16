"""
Linear regression implementations.

This module provides functions for ordinary least squares (OLS),
ridge regression (L2), LASSO regression (L1), and gradient descent-based fitting.
"""

import numpy as np
from scipy import optimize


def ols_fit(X, y):
    """
    Fit ordinary least squares (OLS) linear regression.
    
    Solves: w = (X^T X)^(-1) X^T y
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    y : array (n_samples,)
        Target values
    
    Returns:
    --------
    w : array (n_features,)
        Fitted weights
    
    Notes:
    ------
    This is the closed-form solution to minimize ||y - Xw||^2
    """
    # Add bias term (intercept)
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    
    # Compute (X^T X)^(-1) X^T y
    w = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
    
    return w


def ridge_fit(X, y, lambda_reg=1.0):
    """
    Fit ridge regression (L2 regularization).
    
    Solves: w = (X^T X + λI)^(-1) X^T y
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    y : array (n_samples,)
        Target values
    lambda_reg : float
        Regularization parameter λ
    
    Returns:
    --------
    w : array (n_features,)
        Fitted weights
    
    Notes:
    ------
    Ridge regression adds L2 penalty: ||y - Xw||^2 + λ||w||^2
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    n_features = X_with_bias.shape[1]
    
    # Compute (X^T X + λI)^(-1) X^T y
    # Don't regularize the bias term
    I = np.eye(n_features)
    I[0, 0] = 0  # Don't penalize intercept
    
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    
    w = np.linalg.solve(XtX + lambda_reg * I, Xty)
    
    return w


def lasso_fit(X, y, lambda_reg=1.0):
    """
    Fit LASSO regression (L1 regularization).
    
    Solves: min ||y - Xw||^2 + λ||w||_1
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    y : array (n_samples,)
        Target values
    lambda_reg : float
        Regularization parameter λ
    
    Returns:
    --------
    w : array (n_features,)
        Fitted weights
    
    Notes:
    ------
    LASSO uses L1 penalty which promotes sparsity.
    No closed-form solution; uses coordinate descent.
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    n_features = X_with_bias.shape[1]
    
    # Objective function
    def objective(w):
        residuals = y - X_with_bias @ w
        mse = np.mean(residuals**2)
        # Don't penalize intercept (first element)
        l1_penalty = lambda_reg * np.sum(np.abs(w[1:]))
        return mse + l1_penalty
    
    # Initial guess
    w0 = np.zeros(n_features)
    
    # Optimize using L-BFGS-B (handles non-differentiability approximately)
    result = optimize.minimize(objective, w0, method='L-BFGS-B')
    
    return result.x


def linear_predict(X, w):
    """
    Make predictions using linear model.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    w : array (n_features + 1,)
        Weights (includes intercept as first element)
    
    Returns:
    --------
    y_pred : array (n_samples,)
        Predictions
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    return X_with_bias @ w


def linear_regression_gd(X, y, learning_rate=0.01, n_iterations=1000, 
                         lambda_reg=0.0, reg_type='none'):
    """
    Fit linear regression using gradient descent.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    y : array (n_samples,)
        Target values
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Number of iterations
    lambda_reg : float
        Regularization parameter
    reg_type : str
        'none', 'ridge', or 'lasso'
    
    Returns:
    --------
    dict
        Contains 'weights', 'loss_history', 'weight_history'
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])
    n_samples, n_features = X_with_bias.shape
    
    # Initialize weights
    w = np.zeros(n_features)
    
    # History tracking
    loss_history = []
    weight_history = [w.copy()]
    
    for iteration in range(n_iterations):
        # Predictions
        y_pred = X_with_bias @ w
        
        # Compute loss (MSE)
        residuals = y_pred - y
        loss = np.mean(residuals**2)
        
        # Add regularization to loss
        if reg_type == 'ridge':
            loss += lambda_reg * np.sum(w[1:]**2)  # Don't penalize intercept
        elif reg_type == 'lasso':
            loss += lambda_reg * np.sum(np.abs(w[1:]))
        
        loss_history.append(loss)
        
        # Compute gradient
        grad = (2 / n_samples) * X_with_bias.T @ residuals
        
        # Add regularization gradient
        if reg_type == 'ridge':
            grad[1:] += 2 * lambda_reg * w[1:]
        elif reg_type == 'lasso':
            grad[1:] += lambda_reg * np.sign(w[1:])
        
        # Update weights
        w = w - learning_rate * grad
        weight_history.append(w.copy())
    
    return {
        'weights': w,
        'loss_history': loss_history,
        'weight_history': np.array(weight_history)
    }


def compute_pseudoinverse(X):
    """
    Compute Moore-Penrose pseudoinverse.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Matrix
    
    Returns:
    --------
    X_pinv : array (n_features, n_samples)
        Pseudoinverse
    
    Notes:
    ------
    X_pinv = (X^T X)^(-1) X^T
    """
    return np.linalg.pinv(X)


def standardize_features(X):
    """
    Standardize features to zero mean and unit variance.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    
    Returns:
    --------
    X_std : array (n_samples, n_features)
        Standardized features
    mean : array (n_features,)
        Feature means
    std : array (n_features,)
        Feature standard deviations
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    X_std = (X - mean) / std
    
    return X_std, mean, std


def normalize_features(X):
    """
    Normalize features to [0, 1] range.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Feature matrix
    
    Returns:
    --------
    X_norm : array (n_samples, n_features)
        Normalized features
    min_vals : array (n_features,)
        Minimum values
    max_vals : array (n_features,)
        Maximum values
    """
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    X_norm = (X - min_vals) / range_vals
    
    return X_norm, min_vals, max_vals
