"""
Loss functions with regularization for optimization demonstrations.

This module provides MSE, cross-entropy, and regularized loss functions.
"""

import numpy as np


def mse_loss(theta, X, y):
    """
    Mean Squared Error loss.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features (n_samples, n_features)
    y : array-like
        Target values (n_samples,)
    
    Returns:
    --------
    float
        MSE loss
    """
    predictions = X @ theta
    return np.mean((y - predictions) ** 2)


def mse_gradient(theta, X, y):
    """
    Gradient of MSE loss.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    
    Returns:
    --------
    array
        Gradient ∇L(theta)
    """
    n = len(y)
    predictions = X @ theta
    return -2 / n * X.T @ (y - predictions)


def mse_with_l1_reg(theta, X, y, lambda_reg=0.1):
    """
    MSE loss with L1 (LASSO) regularization.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_reg : float
        Regularization strength λ
    
    Returns:
    --------
    float
        Regularized loss
    
    Formula:
    --------
    L(θ) = MSE + λ * ||θ||₁
    """
    mse = mse_loss(theta, X, y)
    l1_penalty = lambda_reg * np.sum(np.abs(theta))
    return mse + l1_penalty


def mse_with_l1_gradient(theta, X, y, lambda_reg=0.1):
    """
    Gradient of MSE with L1 regularization.
    
    Note: L1 is non-differentiable at 0, we use sign function.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_reg : float
        Regularization strength
    
    Returns:
    --------
    array
        Gradient
    """
    mse_grad = mse_gradient(theta, X, y)
    l1_grad = lambda_reg * np.sign(theta)
    return mse_grad + l1_grad


def mse_with_l2_reg(theta, X, y, lambda_reg=0.1):
    """
    MSE loss with L2 (Ridge) regularization.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_reg : float
        Regularization strength λ
    
    Returns:
    --------
    float
        Regularized loss
    
    Formula:
    --------
    L(θ) = MSE + λ * ||θ||₂²
    """
    mse = mse_loss(theta, X, y)
    l2_penalty = lambda_reg * np.sum(theta ** 2)
    return mse + l2_penalty


def mse_with_l2_gradient(theta, X, y, lambda_reg=0.1):
    """
    Gradient of MSE with L2 regularization.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_reg : float
        Regularization strength
    
    Returns:
    --------
    array
        Gradient
    """
    mse_grad = mse_gradient(theta, X, y)
    l2_grad = 2 * lambda_reg * theta
    return mse_grad + l2_grad


def mse_with_elastic_net(theta, X, y, lambda_l1=0.1, lambda_l2=0.1):
    """
    MSE loss with Elastic Net regularization (L1 + L2).
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_l1 : float
        L1 regularization strength
    lambda_l2 : float
        L2 regularization strength
    
    Returns:
    --------
    float
        Regularized loss
    
    Formula:
    --------
    L(θ) = MSE + λ₁ * ||θ||₁ + λ₂ * ||θ||₂²
    """
    mse = mse_loss(theta, X, y)
    l1_penalty = lambda_l1 * np.sum(np.abs(theta))
    l2_penalty = lambda_l2 * np.sum(theta ** 2)
    return mse + l1_penalty + l2_penalty


def mse_with_elastic_net_gradient(theta, X, y, lambda_l1=0.1, lambda_l2=0.1):
    """
    Gradient of MSE with Elastic Net regularization.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Target values
    lambda_l1 : float
        L1 regularization strength
    lambda_l2 : float
        L2 regularization strength
    
    Returns:
    --------
    array
        Gradient
    """
    mse_grad = mse_gradient(theta, X, y)
    l1_grad = lambda_l1 * np.sign(theta)
    l2_grad = 2 * lambda_l2 * theta
    return mse_grad + l1_grad + l2_grad


def cross_entropy_loss(theta, X, y):
    """
    Binary cross-entropy loss for logistic regression.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Binary labels (0 or 1)
    
    Returns:
    --------
    float
        Cross-entropy loss
    
    Formula:
    --------
    L(θ) = -1/n * Σ[y*log(σ(Xθ)) + (1-y)*log(1-σ(Xθ))]
    where σ is the sigmoid function
    """
    n = len(y)
    z = X @ theta
    # Sigmoid function
    sigmoid = 1 / (1 + np.exp(-z))
    # Clip to avoid log(0)
    sigmoid = np.clip(sigmoid, 1e-10, 1 - 1e-10)
    loss = -1/n * np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid))
    return loss


def cross_entropy_gradient(theta, X, y):
    """
    Gradient of binary cross-entropy loss.
    
    Parameters:
    -----------
    theta : array-like
        Model parameters
    X : array-like
        Input features
    y : array-like
        Binary labels
    
    Returns:
    --------
    array
        Gradient
    """
    n = len(y)
    z = X @ theta
    sigmoid = 1 / (1 + np.exp(-z))
    return 1/n * X.T @ (sigmoid - y)


def quadratic_bowl(theta):
    """
    Simple quadratic bowl function for visualization.
    
    Parameters:
    -----------
    theta : array-like
        Parameters [x, y]
    
    Returns:
    --------
    float
        Function value
    
    Formula:
    --------
    f(x, y) = x² + y²
    """
    return np.sum(theta ** 2)


def rosenbrock(theta, a=1, b=100):
    """
    Rosenbrock function (non-convex, banana-shaped valley).
    
    Parameters:
    -----------
    theta : array-like
        Parameters [x, y]
    a : float
        Parameter a (default 1)
    b : float
        Parameter b (default 100)
    
    Returns:
    --------
    float
        Function value
    
    Formula:
    --------
    f(x, y) = (a - x)² + b(y - x²)²
    """
    x, y = theta[0], theta[1]
    return (a - x)**2 + b * (y - x**2)**2


def himmelblau(theta):
    """
    Himmelblau function (multi-modal, 4 local minima).
    
    Parameters:
    -----------
    theta : array-like
        Parameters [x, y]
    
    Returns:
    --------
    float
        Function value
    
    Formula:
    --------
    f(x, y) = (x² + y - 11)² + (x + y² - 7)²
    """
    x, y = theta[0], theta[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def rastrigin(theta, A=10):
    """
    Rastrigin function (highly multi-modal).
    
    Parameters:
    -----------
    theta : array-like
        Parameters
    A : float
        Amplitude parameter
    
    Returns:
    --------
    float
        Function value
    
    Formula:
    --------
    f(x) = A*n + Σ[xᵢ² - A*cos(2πxᵢ)]
    """
    n = len(theta)
    return A * n + np.sum(theta**2 - A * np.cos(2 * np.pi * theta))
