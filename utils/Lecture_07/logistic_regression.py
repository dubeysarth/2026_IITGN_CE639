"""
Logistic Regression Core Functions

This module provides fundamental logistic regression utilities for
binary and multi-class classification, including activation functions,
loss functions, and gradient descent training.
"""

import numpy as np


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Parameters:
    -----------
    z : array_like
        Linear combination of inputs (w^T x + b)
    
    Returns:
    --------
    array_like
        Sigmoid activation σ(z) = 1 / (1 + exp(-z))
    
    Notes:
    ------
    - Maps any real value to [0, 1]
    - Used for binary classification probabilities
    - Numerically stable implementation for large |z|
    """
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Softmax activation function for multi-class classification.
    
    Parameters:
    -----------
    z : array_like, shape (n_samples, n_classes)
        Linear combinations for each class
    
    Returns:
    --------
    array_like, shape (n_samples, n_classes)
        Class probabilities (each row sums to 1)
    
    Notes:
    ------
    - Generalizes sigmoid to K > 2 classes
    - Numerically stable implementation (subtract max)
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Binary cross-entropy loss.
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred : array_like, shape (n_samples,)
        Predicted probabilities [0, 1]
    epsilon : float
        Small constant to prevent log(0)
    
    Returns:
    --------
    float
        Average cross-entropy loss
    
    Notes:
    ------
    L = -(y log(ŷ) + (1-y) log(1-ŷ))
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute loss
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def categorical_cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Categorical cross-entropy loss for multi-class classification.
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples, n_classes)
        One-hot encoded true labels
    y_pred : array_like, shape (n_samples, n_classes)
        Predicted class probabilities
    epsilon : float
        Small constant to prevent log(0)
    
    Returns:
    --------
    float
        Average categorical cross-entropy loss
    
    Notes:
    ------
    L = -Σ y_k log(ŷ_k)
    """
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute loss
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return np.mean(loss)


def logistic_regression_gd(X, y, learning_rate=0.01, n_iterations=1000, 
                          lambda_reg=0.0, verbose=False):
    """
    Train logistic regression using gradient descent.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Training features
    y : array_like, shape (n_samples,)
        Binary labels (0 or 1)
    learning_rate : float
        Learning rate (step size)
    n_iterations : int
        Number of gradient descent iterations
    lambda_reg : float
        L2 regularization strength
    verbose : bool
        Print progress every 100 iterations
    
    Returns:
    --------
    dict
        Contains:
        - 'weights': Trained weights (including bias as first element)
        - 'loss_history': Loss at each iteration
        - 'accuracy_history': Accuracy at each iteration
    
    Notes:
    ------
    Update rule: w = w - α ∇L(w)
    Gradient: ∇L = (1/n) X^T (σ(Xw) - y) + λw
    """
    n_samples, n_features = X.shape
    
    # Add bias term
    X_bias = np.column_stack([np.ones(n_samples), X])
    
    # Initialize weights
    weights = np.zeros(n_features + 1)
    
    # Track history
    loss_history = []
    accuracy_history = []
    
    for iteration in range(n_iterations):
        # Forward pass
        z = X_bias @ weights
        y_pred = sigmoid(z)
        
        # Compute loss
        loss = cross_entropy_loss(y, y_pred)
        
        # Add L2 regularization (don't regularize bias)
        if lambda_reg > 0:
            reg_term = (lambda_reg / (2 * n_samples)) * np.sum(weights[1:]**2)
            loss += reg_term
        
        loss_history.append(loss)
        
        # Compute accuracy
        y_pred_class = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(y_pred_class == y)
        accuracy_history.append(accuracy)
        
        # Compute gradient
        error = y_pred - y
        gradient = (1 / n_samples) * (X_bias.T @ error)
        
        # Add regularization gradient (don't regularize bias)
        if lambda_reg > 0:
            reg_gradient = np.zeros_like(weights)
            reg_gradient[1:] = (lambda_reg / n_samples) * weights[1:]
            gradient += reg_gradient
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Verbose output
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}: "
                  f"Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    return {
        'weights': weights,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history
    }


def logistic_predict(X, weights):
    """
    Predict probabilities using trained logistic regression model.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Input features
    weights : array_like, shape (n_features + 1,)
        Trained weights (including bias as first element)
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Predicted probabilities [0, 1]
    """
    n_samples = X.shape[0]
    X_bias = np.column_stack([np.ones(n_samples), X])
    z = X_bias @ weights
    return sigmoid(z)


def logistic_predict_class(X, weights, threshold=0.5):
    """
    Predict class labels using trained logistic regression model.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Input features
    weights : array_like, shape (n_features + 1,)
        Trained weights (including bias as first element)
    threshold : float
        Decision threshold (default 0.5)
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Predicted class labels (0 or 1)
    """
    probabilities = logistic_predict(X, weights)
    return (probabilities >= threshold).astype(int)


def softmax_regression_gd(X, y, n_classes, learning_rate=0.01, 
                         n_iterations=1000, lambda_reg=0.0, verbose=False):
    """
    Train softmax regression (multi-class) using gradient descent.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Training features
    y : array_like, shape (n_samples,)
        Class labels (0 to n_classes-1)
    n_classes : int
        Number of classes
    learning_rate : float
        Learning rate
    n_iterations : int
        Number of iterations
    lambda_reg : float
        L2 regularization strength
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict
        Contains:
        - 'weights': shape (n_features + 1, n_classes)
        - 'loss_history': Loss at each iteration
        - 'accuracy_history': Accuracy at each iteration
    """
    n_samples, n_features = X.shape
    
    # Add bias term
    X_bias = np.column_stack([np.ones(n_samples), X])
    
    # Initialize weights (one set per class)
    weights = np.zeros((n_features + 1, n_classes))
    
    # One-hot encode labels
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y] = 1
    
    # Track history
    loss_history = []
    accuracy_history = []
    
    for iteration in range(n_iterations):
        # Forward pass
        z = X_bias @ weights
        y_pred = softmax(z)
        
        # Compute loss
        loss = categorical_cross_entropy_loss(y_onehot, y_pred)
        
        # Add L2 regularization
        if lambda_reg > 0:
            reg_term = (lambda_reg / (2 * n_samples)) * np.sum(weights[1:]**2)
            loss += reg_term
        
        loss_history.append(loss)
        
        # Compute accuracy
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == y)
        accuracy_history.append(accuracy)
        
        # Compute gradient
        error = y_pred - y_onehot
        gradient = (1 / n_samples) * (X_bias.T @ error)
        
        # Add regularization gradient
        if lambda_reg > 0:
            reg_gradient = np.zeros_like(weights)
            reg_gradient[1:] = (lambda_reg / n_samples) * weights[1:]
            gradient += reg_gradient
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Verbose output
        if verbose and (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}: "
                  f"Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    return {
        'weights': weights,
        'loss_history': loss_history,
        'accuracy_history': accuracy_history
    }


def softmax_predict(X, weights):
    """
    Predict class probabilities using trained softmax model.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Input features
    weights : array_like, shape (n_features + 1, n_classes)
        Trained weights
    
    Returns:
    --------
    array_like, shape (n_samples, n_classes)
        Predicted class probabilities
    """
    n_samples = X.shape[0]
    X_bias = np.column_stack([np.ones(n_samples), X])
    z = X_bias @ weights
    return softmax(z)


def softmax_predict_class(X, weights):
    """
    Predict class labels using trained softmax model.
    
    Parameters:
    -----------
    X : array_like, shape (n_samples, n_features)
        Input features
    weights : array_like, shape (n_features + 1, n_classes)
        Trained weights
    
    Returns:
    --------
    array_like, shape (n_samples,)
        Predicted class labels
    """
    probabilities = softmax_predict(X, weights)
    return np.argmax(probabilities, axis=1)
