"""
Kernel regression implementations.

This module provides kernel functions and kernel regression (dual form)
for linear, polynomial, and RBF kernels.
"""

import numpy as np


def linear_kernel(X1, X2):
    """
    Compute linear kernel matrix.
    
    k(x, x') = x^T x'
    
    Parameters:
    -----------
    X1 : array (n1, n_features)
        First set of samples
    X2 : array (n2, n_features)
        Second set of samples
    
    Returns:
    --------
    K : array (n1, n2)
        Kernel matrix where K[i,j] = k(X1[i], X2[j])
    """
    return X1 @ X2.T


def polynomial_kernel(X1, X2, degree=2, coef0=1.0):
    """
    Compute polynomial kernel matrix.
    
    k(x, x') = (x^T x' + c)^p
    
    Parameters:
    -----------
    X1 : array (n1, n_features)
        First set of samples
    X2 : array (n2, n_features)
        Second set of samples
    degree : int
        Polynomial degree p
    coef0 : float
        Constant term c
    
    Returns:
    --------
    K : array (n1, n2)
        Kernel matrix
    """
    return (X1 @ X2.T + coef0) ** degree


def rbf_kernel(X1, X2, sigma=1.0):
    """
    Compute RBF (Gaussian) kernel matrix.
    
    k(x, x') = exp(-||x - x'||^2 / (2σ^2))
    
    Parameters:
    -----------
    X1 : array (n1, n_features)
        First set of samples
    X2 : array (n2, n_features)
        Second set of samples
    sigma : float
        Bandwidth parameter σ
    
    Returns:
    --------
    K : array (n1, n2)
        Kernel matrix
    
    Notes:
    ------
    RBF kernel corresponds to infinite-dimensional feature space.
    """
    # Compute pairwise squared distances
    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2x^T x'
    X1_sq = np.sum(X1**2, axis=1, keepdims=True)
    X2_sq = np.sum(X2**2, axis=1, keepdims=True)
    
    sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
    
    # Compute kernel
    K = np.exp(-sq_dists / (2 * sigma**2))
    
    return K


def kernel_matrix(X, kernel_type='linear', **kernel_params):
    """
    Compute kernel matrix for a single dataset.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Input samples
    kernel_type : str
        'linear', 'polynomial', or 'rbf'
    **kernel_params : dict
        Kernel-specific parameters
    
    Returns:
    --------
    K : array (n_samples, n_samples)
        Kernel matrix
    """
    if kernel_type == 'linear':
        return linear_kernel(X, X)
    elif kernel_type == 'polynomial':
        degree = kernel_params.get('degree', 2)
        coef0 = kernel_params.get('coef0', 1.0)
        return polynomial_kernel(X, X, degree=degree, coef0=coef0)
    elif kernel_type == 'rbf':
        sigma = kernel_params.get('sigma', 1.0)
        return rbf_kernel(X, X, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def kernel_regression_fit(X, y, kernel_type='linear', lambda_reg=1.0, **kernel_params):
    """
    Fit kernel regression using dual formulation.
    
    Solves: (K + λI)α = y
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Training features
    y : array (n_samples,)
        Training targets
    kernel_type : str
        'linear', 'polynomial', or 'rbf'
    lambda_reg : float
        Regularization parameter λ
    **kernel_params : dict
        Kernel-specific parameters
    
    Returns:
    --------
    dict
        Model containing 'alpha', 'X_train', 'kernel_type', 'kernel_params', 'lambda_reg'
    
    Notes:
    ------
    Dual solution: ŷ(x) = Σ α_i k(x, x_i)
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Compute kernel matrix
    K = kernel_matrix(X, kernel_type=kernel_type, **kernel_params)
    
    # Solve (K + λI)α = y
    n_samples = len(X)
    alpha = np.linalg.solve(K + lambda_reg * np.eye(n_samples), y)
    
    return {
        'alpha': alpha,
        'X_train': X,
        'kernel_type': kernel_type,
        'kernel_params': kernel_params,
        'lambda_reg': lambda_reg
    }


def kernel_regression_predict(X_test, model):
    """
    Make predictions using kernel regression model.
    
    Parameters:
    -----------
    X_test : array (n_test, n_features)
        Test features
    model : dict
        Model from kernel_regression_fit
    
    Returns:
    --------
    y_pred : array (n_test,)
        Predictions
    """
    # Ensure X_test is 2D
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    # Compute kernel between test and training points
    X_train = model['X_train']
    kernel_type = model['kernel_type']
    kernel_params = model['kernel_params']
    
    if kernel_type == 'linear':
        K_test = linear_kernel(X_test, X_train)
    elif kernel_type == 'polynomial':
        degree = kernel_params.get('degree', 2)
        coef0 = kernel_params.get('coef0', 1.0)
        K_test = polynomial_kernel(X_test, X_train, degree=degree, coef0=coef0)
    elif kernel_type == 'rbf':
        sigma = kernel_params.get('sigma', 1.0)
        K_test = rbf_kernel(X_test, X_train, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Predict: ŷ = K_test @ α
    y_pred = K_test @ model['alpha']
    
    return y_pred


def primal_to_dual_weights(w, X, kernel_type='linear', **kernel_params):
    """
    Convert primal weights to dual coefficients.
    
    For linear kernel: w = X^T α
    
    Parameters:
    -----------
    w : array (n_features,)
        Primal weights
    X : array (n_samples, n_features)
        Training data
    kernel_type : str
        Kernel type
    **kernel_params : dict
        Kernel parameters
    
    Returns:
    --------
    alpha : array (n_samples,)
        Dual coefficients
    
    Notes:
    ------
    Only exact for linear kernel. For nonlinear kernels,
    this is an approximation.
    """
    if kernel_type == 'linear':
        # Solve X^T α = w for α
        # This is underdetermined if n_samples > n_features
        alpha = np.linalg.lstsq(X.T, w, rcond=None)[0]
        return alpha
    else:
        raise NotImplementedError("Primal to dual conversion only implemented for linear kernel")


def dual_to_primal_weights(alpha, X, kernel_type='linear'):
    """
    Convert dual coefficients to primal weights.
    
    For linear kernel: w = X^T α
    
    Parameters:
    -----------
    alpha : array (n_samples,)
        Dual coefficients
    X : array (n_samples, n_features)
        Training data
    kernel_type : str
        Kernel type
    
    Returns:
    --------
    w : array (n_features,)
        Primal weights
    
    Notes:
    ------
    Only valid for linear kernel. Nonlinear kernels don't have
    explicit primal weights.
    """
    if kernel_type == 'linear':
        w = X.T @ alpha
        return w
    else:
        raise ValueError("Primal weights don't exist for nonlinear kernels")


def kernel_ridge_regression(X, y, kernel_type='rbf', lambda_reg=1.0, **kernel_params):
    """
    Kernel ridge regression (convenience wrapper).
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Training features
    y : array (n_samples,)
        Training targets
    kernel_type : str
        Kernel type
    lambda_reg : float
        Regularization parameter
    **kernel_params : dict
        Kernel parameters
    
    Returns:
    --------
    dict
        Fitted model
    """
    return kernel_regression_fit(X, y, kernel_type=kernel_type, 
                                lambda_reg=lambda_reg, **kernel_params)


def compute_kernel_gram_matrix(X, kernel_func, **kernel_params):
    """
    Compute Gram matrix (kernel matrix) using custom kernel function.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Input data
    kernel_func : callable
        Kernel function k(x1, x2, **params)
    **kernel_params : dict
        Parameters for kernel function
    
    Returns:
    --------
    K : array (n_samples, n_samples)
        Gram matrix
    """
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j], **kernel_params)
    
    return K


def kernel_pca_transform(X, n_components=2, kernel_type='rbf', **kernel_params):
    """
    Kernel PCA transformation.
    
    Parameters:
    -----------
    X : array (n_samples, n_features)
        Input data
    n_components : int
        Number of components
    kernel_type : str
        Kernel type
    **kernel_params : dict
        Kernel parameters
    
    Returns:
    --------
    X_transformed : array (n_samples, n_components)
        Transformed data
    """
    # Compute kernel matrix
    K = kernel_matrix(X, kernel_type=kernel_type, **kernel_params)
    
    # Center kernel matrix
    n = len(K)
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    X_transformed = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
    
    return X_transformed
