"""
Calculus operations for derivatives, gradients, and Jacobians.

This module provides numerical computation of derivatives, partial derivatives,
gradients, Jacobians, and Hessians.
"""

import numpy as np


def derivative(f, x, h=1e-5):
    """
    Compute numerical derivative using central difference.
    
    Parameters:
    -----------
    f : callable
        Function f(x) returning scalar
    x : float
        Point at which to compute derivative
    h : float
        Step size
    
    Returns:
    --------
    float
        Derivative f'(x)
    
    Formula:
    --------
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def partial_derivative(f, x, i, h=1e-5):
    """
    Compute partial derivative with respect to x_i.
    
    Parameters:
    -----------
    f : callable
        Function f(x) where x is array-like
    x : array-like
        Point at which to compute partial
    i : int
        Index of variable
    h : float
        Step size
    
    Returns:
    --------
    float
        Partial derivative ∂f/∂x_i
    """
    x = np.array(x, dtype=float)
    x_plus = x.copy()
    x_minus = x.copy()
    
    x_plus[i] += h
    x_minus[i] -= h
    
    return (f(x_plus) - f(x_minus)) / (2 * h)


def gradient(f, x, h=1e-5):
    """
    Compute gradient vector ∇f(x).
    
    Parameters:
    -----------
    f : callable
        Scalar function f(x) where x is array-like
    x : array-like
        Point at which to compute gradient
    h : float
        Step size
    
    Returns:
    --------
    array
        Gradient vector [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]
    
    Formula:
    --------
    ∇f(x) = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]^T
    """
    x = np.array(x, dtype=float)
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        grad[i] = partial_derivative(f, x, i, h)
    
    return grad


def jacobian(f, x, h=1e-5):
    """
    Compute Jacobian matrix J_f(x).
    
    Parameters:
    -----------
    f : callable
        Vector function f(x) returning array-like
    x : array-like
        Point at which to compute Jacobian
    h : float
        Step size
    
    Returns:
    --------
    array
        Jacobian matrix (m × n) where m = len(f(x)), n = len(x)
        J[i,j] = ∂f_i/∂x_j
    
    Formula:
    --------
    J = [∂f_i/∂x_j] for i=1..m, j=1..n
    """
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Evaluate function to get output dimension
    f_x = np.array(f(x))
    m = len(f_x) if f_x.ndim > 0 else 1
    
    J = np.zeros((m, n))
    
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[j] += h
        x_minus[j] -= h
        
        f_plus = np.array(f(x_plus))
        f_minus = np.array(f(x_minus))
        
        J[:, j] = (f_plus - f_minus) / (2 * h)
    
    return J


def hessian(f, x, h=1e-5):
    """
    Compute Hessian matrix H_f(x) (matrix of second derivatives).
    
    Parameters:
    -----------
    f : callable
        Scalar function f(x)
    x : array-like
        Point at which to compute Hessian
    h : float
        Step size
    
    Returns:
    --------
    array
        Hessian matrix (n × n)
        H[i,j] = ∂²f/∂x_i∂x_j
    
    Formula:
    --------
    H[i,j] = ∂²f/∂x_i∂x_j
    """
    x = np.array(x, dtype=float)
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Compute mixed partial using finite differences
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h
            x_pp[j] += h
            
            x_pm[i] += h
            x_pm[j] -= h
            
            x_mp[i] -= h
            x_mp[j] += h
            
            x_mm[i] -= h
            x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
    
    return H


def directional_derivative(f, x, v, h=1e-5):
    """
    Compute directional derivative D_v f(x).
    
    Parameters:
    -----------
    f : callable
        Scalar function f(x)
    x : array-like
        Point at which to compute directional derivative
    v : array-like
        Direction vector (will be normalized)
    h : float
        Step size
    
    Returns:
    --------
    float
        Directional derivative D_v f(x) = ∇f(x) · v
    
    Formula:
    --------
    D_v f(x) = ∇f(x)^T · v/||v||
    """
    x = np.array(x, dtype=float)
    v = np.array(v, dtype=float)
    
    # Normalize direction
    v_norm = v / np.linalg.norm(v)
    
    # Compute gradient
    grad = gradient(f, x, h)
    
    # Directional derivative is dot product
    return np.dot(grad, v_norm)


def gradient_descent_step(f, x, learning_rate=0.01, h=1e-5):
    """
    Perform one gradient descent step.
    
    Parameters:
    -----------
    f : callable
        Function to minimize
    x : array-like
        Current point
    learning_rate : float
        Step size
    h : float
        Finite difference step
    
    Returns:
    --------
    array
        Updated point x - η∇f(x)
    """
    x = np.array(x, dtype=float)
    grad = gradient(f, x, h)
    return x - learning_rate * grad


def check_gradient(f, grad_f, x, h=1e-5, tol=1e-4):
    """
    Check analytical gradient against numerical gradient.
    
    Parameters:
    -----------
    f : callable
        Function
    grad_f : callable
        Analytical gradient function
    x : array-like
        Point to check
    h : float
        Finite difference step
    tol : float
        Tolerance for comparison
    
    Returns:
    --------
    dict
        Contains 'numerical', 'analytical', 'difference', 'passed'
    """
    x = np.array(x, dtype=float)
    
    numerical = gradient(f, x, h)
    analytical = np.array(grad_f(x))
    
    difference = np.linalg.norm(numerical - analytical)
    passed = difference < tol
    
    return {
        'numerical': numerical,
        'analytical': analytical,
        'difference': difference,
        'passed': passed
    }
