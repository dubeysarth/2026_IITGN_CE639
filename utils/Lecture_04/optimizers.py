"""
Optimization algorithms for gradient descent and variants.

This module provides implementations of gradient descent, SGD, Adam,
and learning rate schedules.
"""

import numpy as np


def gradient_descent(loss_fn, grad_fn, theta_init, learning_rate=0.01, 
                    max_iters=100, tol=1e-6, store_history=True):
    """
    Basic gradient descent optimization.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function L(theta)
    grad_fn : callable
        Gradient function ∇L(theta)
    theta_init : array-like
        Initial parameters
    learning_rate : float
        Learning rate η
    max_iters : int
        Maximum iterations
    tol : float
        Convergence tolerance
    store_history : bool
        Store theta and loss history
    
    Returns:
    --------
    dict
        Contains 'theta', 'loss', 'history' (if store_history=True)
    """
    theta = np.array(theta_init, dtype=float)
    
    if store_history:
        theta_history = [theta.copy()]
        loss_history = [loss_fn(theta)]
    
    for i in range(max_iters):
        # Compute gradient
        grad = grad_fn(theta)
        
        # Update parameters
        theta_new = theta - learning_rate * grad
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            theta = theta_new
            if store_history:
                theta_history.append(theta.copy())
                loss_history.append(loss_fn(theta))
            break
        
        theta = theta_new
        
        if store_history:
            theta_history.append(theta.copy())
            loss_history.append(loss_fn(theta))
    
    result = {
        'theta': theta,
        'loss': loss_fn(theta),
        'iterations': i + 1
    }
    
    if store_history:
        result['theta_history'] = np.array(theta_history)
        result['loss_history'] = np.array(loss_history)
    
    return result


def stochastic_gradient_descent(loss_fn, grad_fn, theta_init, X, y,
                                batch_size=32, learning_rate=0.01,
                                max_epochs=100, store_history=True):
    """
    Stochastic gradient descent with minibatches.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function L(theta, X, y)
    grad_fn : callable
        Gradient function ∇L(theta, X, y)
    theta_init : array-like
        Initial parameters
    X : array-like
        Input data
    y : array-like
        Target data
    batch_size : int
        Minibatch size
    learning_rate : float
        Learning rate
    max_epochs : int
        Maximum epochs
    store_history : bool
        Store history
    
    Returns:
    --------
    dict
        Contains 'theta', 'loss', 'history'
    """
    theta = np.array(theta_init, dtype=float)
    n_samples = len(X)
    
    if store_history:
        theta_history = [theta.copy()]
        loss_history = [loss_fn(theta, X, y)]
    
    for epoch in range(max_epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Minibatch updates
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradient on minibatch
            grad = grad_fn(theta, X_batch, y_batch)
            
            # Update
            theta = theta - learning_rate * grad
            
            if store_history:
                theta_history.append(theta.copy())
                loss_history.append(loss_fn(theta, X, y))
    
    result = {
        'theta': theta,
        'loss': loss_fn(theta, X, y),
        'epochs': max_epochs
    }
    
    if store_history:
        result['theta_history'] = np.array(theta_history)
        result['loss_history'] = np.array(loss_history)
    
    return result


def adam_optimizer(loss_fn, grad_fn, theta_init, learning_rate=0.001,
                  beta1=0.9, beta2=0.999, epsilon=1e-8,
                  max_iters=100, store_history=True):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function
    grad_fn : callable
        Gradient function
    theta_init : array-like
        Initial parameters
    learning_rate : float
        Learning rate α
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    epsilon : float
        Small constant for numerical stability
    max_iters : int
        Maximum iterations
    store_history : bool
        Store history
    
    Returns:
    --------
    dict
        Contains 'theta', 'loss', 'history'
    """
    theta = np.array(theta_init, dtype=float)
    m = np.zeros_like(theta)  # First moment
    v = np.zeros_like(theta)  # Second moment
    
    if store_history:
        theta_history = [theta.copy()]
        loss_history = [loss_fn(theta)]
    
    for t in range(1, max_iters + 1):
        # Compute gradient
        grad = grad_fn(theta)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** t)
        
        # Update parameters
        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if store_history:
            theta_history.append(theta.copy())
            loss_history.append(loss_fn(theta))
    
    result = {
        'theta': theta,
        'loss': loss_fn(theta),
        'iterations': max_iters
    }
    
    if store_history:
        result['theta_history'] = np.array(theta_history)
        result['loss_history'] = np.array(loss_history)
    
    return result


def compute_gradient(f, theta, epsilon=1e-5):
    """
    Compute numerical gradient using finite differences.
    
    Parameters:
    -----------
    f : callable
        Function to differentiate
    theta : array-like
        Point at which to compute gradient
    epsilon : float
        Step size for finite differences
    
    Returns:
    --------
    array
        Numerical gradient
    """
    theta = np.array(theta, dtype=float)
    grad = np.zeros_like(theta)
    
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        
        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        
        grad[i] = (f(theta_plus) - f(theta_minus)) / (2 * epsilon)
    
    return grad


def learning_rate_schedule(schedule_type, initial_lr, iteration, **kwargs):
    """
    Learning rate schedules.
    
    Parameters:
    -----------
    schedule_type : str
        'constant', 'step', 'exponential', 'cosine'
    initial_lr : float
        Initial learning rate
    iteration : int
        Current iteration
    **kwargs : dict
        Schedule-specific parameters
    
    Returns:
    --------
    float
        Learning rate at current iteration
    """
    if schedule_type == 'constant':
        return initial_lr
    
    elif schedule_type == 'step':
        # Step decay: lr = initial_lr * decay_rate^(iteration // step_size)
        step_size = kwargs.get('step_size', 10)
        decay_rate = kwargs.get('decay_rate', 0.5)
        return initial_lr * (decay_rate ** (iteration // step_size))
    
    elif schedule_type == 'exponential':
        # Exponential decay: lr = initial_lr * exp(-decay_rate * iteration)
        decay_rate = kwargs.get('decay_rate', 0.01)
        return initial_lr * np.exp(-decay_rate * iteration)
    
    elif schedule_type == 'cosine':
        # Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * iteration / max_iters))
        max_iters = kwargs.get('max_iters', 100)
        min_lr = kwargs.get('min_lr', 0)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * iteration / max_iters))
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def gradient_descent_with_momentum(loss_fn, grad_fn, theta_init, learning_rate=0.01,
                                   momentum=0.9, max_iters=100, store_history=True):
    """
    Gradient descent with momentum.
    
    Parameters:
    -----------
    loss_fn : callable
        Loss function
    grad_fn : callable
        Gradient function
    theta_init : array-like
        Initial parameters
    learning_rate : float
        Learning rate
    momentum : float
        Momentum coefficient (0 to 1)
    max_iters : int
        Maximum iterations
    store_history : bool
        Store history
    
    Returns:
    --------
    dict
        Contains 'theta', 'loss', 'history'
    """
    theta = np.array(theta_init, dtype=float)
    velocity = np.zeros_like(theta)
    
    if store_history:
        theta_history = [theta.copy()]
        loss_history = [loss_fn(theta)]
    
    for i in range(max_iters):
        # Compute gradient
        grad = grad_fn(theta)
        
        # Update velocity
        velocity = momentum * velocity - learning_rate * grad
        
        # Update parameters
        theta = theta + velocity
        
        if store_history:
            theta_history.append(theta.copy())
            loss_history.append(loss_fn(theta))
    
    result = {
        'theta': theta,
        'loss': loss_fn(theta),
        'iterations': max_iters
    }
    
    if store_history:
        result['theta_history'] = np.array(theta_history)
        result['loss_history'] = np.array(loss_history)
    
    return result
