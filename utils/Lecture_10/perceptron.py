"""
Perceptron and single-neuron utilities for Lecture 10: Feedforward Neural Networks.

Implements the simplest building block of a neural network from scratch using NumPy,
including the forward pass, decision boundary, and one gradient-descent step.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core Perceptron Operations
# ─────────────────────────────────────────────────────────────────────────────

def perceptron_forward(
    x: np.ndarray,
    w: np.ndarray,
    b: float,
    activation: str = "sigmoid"
) -> Tuple[float, float]:
    """
    Compute the forward pass for a single perceptron.

    Parameters
    ----------
    x : np.ndarray
        Input vector, shape (n_features,)
    w : np.ndarray
        Weight vector, shape (n_features,)
    b : float
        Bias scalar
    activation : str
        One of 'sigmoid', 'tanh', 'relu', 'linear', 'step'

    Returns
    -------
    z : float
        Pre-activation (linear combination)
    a : float
        Post-activation (output)

    Notes
    -----
    z = w^T x + b, then a = g(z) where g is the activation function.
    A single perceptron with sigmoid activation is exactly logistic regression.
    """
    z = float(np.dot(w, x) + b)

    if activation == "sigmoid":
        a = 1.0 / (1.0 + np.exp(-z))
    elif activation == "tanh":
        a = float(np.tanh(z))
    elif activation == "relu":
        a = max(0.0, z)
    elif activation == "leaky_relu":
        a = z if z > 0 else 0.01 * z
    elif activation == "linear":
        a = z
    elif activation == "step":
        a = 1.0 if z >= 0 else 0.0
    else:
        raise ValueError(f"Unknown activation: '{activation}'. "
                         f"Choose from sigmoid, tanh, relu, leaky_relu, linear, step.")

    return z, a


def perceptron_decision_boundary(
    w: np.ndarray,
    b: float,
    x_range: Tuple[float, float] = (-3, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the decision boundary line for a 2-D perceptron.

    The boundary is at z = w[0]*x1 + w[1]*x2 + b = 0,
    which rearranges to x2 = -(w[0]*x1 + b) / w[1].

    Parameters
    ----------
    w : np.ndarray
        Weight vector, shape (2,)
    b : float
        Bias
    x_range : Tuple[float, float]
        Range of x1 values for the plot

    Returns
    -------
    x1_vals : np.ndarray
        x1 coordinates of the boundary
    x2_vals : np.ndarray
        Corresponding x2 coordinates

    Raises
    ------
    ValueError
        If w[1] is zero (vertical boundary — not representable as x2=f(x1)).
    """
    if len(w) != 2:
        raise ValueError("Decision boundary requires exactly 2-D weights.")
    if abs(w[1]) < 1e-10:
        raise ValueError("w[1] ≈ 0: boundary is vertical. Use a mesh instead.")

    x1_vals = np.linspace(x_range[0], x_range[1], 300)
    x2_vals = -(w[0] * x1_vals + b) / w[1]
    return x1_vals, x2_vals


def neuron_activation_region(
    w: np.ndarray,
    b: float,
    grid_size: int = 200,
    x_range: Tuple[float, float] = (-3, 3)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ON/OFF activation region for a single ReLU neuron over a 2-D grid.

    Returns meshgrid arrays suitable for plt.contourf().

    Parameters
    ----------
    w : np.ndarray
        Weight vector, shape (2,)
    b : float
        Bias
    grid_size : int
        Grid resolution
    x_range : Tuple[float, float]
        Symmetric range for both axes

    Returns
    -------
    X1, X2 : np.ndarray
        Meshgrid coordinate arrays
    Z : np.ndarray
        Pre-activation values (positive = neuron is ON)
    """
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    X1, X2 = np.meshgrid(xs, xs)
    Z = w[0] * X1 + w[1] * X2 + b
    return X1, X2, Z


# ─────────────────────────────────────────────────────────────────────────────
# Single Neuron Gradient Descent
# ─────────────────────────────────────────────────────────────────────────────

def single_neuron_gradient_step(
    x: float,
    y: float,
    w: float,
    b: float,
    lr: float = 0.1
) -> Dict[str, float]:
    """
    Perform one gradient descent step for a single linear neuron (regression).

    This replicates Practice Problem 3 from the slides exactly.

    Model:  ŷ = w*x + b
    Loss:   L = 0.5 * (ŷ - y)²

    Gradients:
        ∂L/∂ŷ = (ŷ - y)
        ∂L/∂w = (ŷ - y) * x
        ∂L/∂b = (ŷ - y)

    Parameters
    ----------
    x : float
        Single input feature
    y : float
        True label
    w : float
        Current weight
    b : float
        Current bias
    lr : float
        Learning rate

    Returns
    -------
    dict with keys: y_hat, loss, grad_w, grad_b, w_new, b_new, y_hat_new
    """
    # Forward pass
    y_hat = w * x + b
    loss = 0.5 * (y_hat - y) ** 2

    # Gradients
    d_loss_d_yhat = y_hat - y
    grad_w = d_loss_d_yhat * x
    grad_b = d_loss_d_yhat * 1.0

    # Update
    w_new = w - lr * grad_w
    b_new = b - lr * grad_b
    y_hat_new = w_new * x + b_new

    return {
        "y_hat": y_hat,
        "loss": loss,
        "grad_w": grad_w,
        "grad_b": grad_b,
        "w_new": w_new,
        "b_new": b_new,
        "y_hat_new": y_hat_new,
    }


def single_neuron_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: float = 0.0,
    b_init: float = 0.0,
    lr: float = 0.01,
    epochs: int = 100
) -> Dict[str, List]:
    """
    Full gradient descent for a single linear neuron over multiple epochs.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n,) or (n, 1)
    y : np.ndarray
        Targets, shape (n,)
    w_init : float
        Initial weight
    b_init : float
        Initial bias
    lr : float
        Learning rate
    epochs : int
        Number of full-data passes

    Returns
    -------
    history : dict
        'loss', 'w', 'b' lists across epochs
    """
    X = X.ravel()
    y = y.ravel()
    n = len(X)
    w, b = w_init, b_init

    history = {"loss": [], "w": [w], "b": [b]}

    for _ in range(epochs):
        # Full-batch gradient
        y_hat = w * X + b
        loss = float(np.mean(0.5 * (y_hat - y) ** 2))
        grad_w = float(np.mean((y_hat - y) * X))
        grad_b = float(np.mean(y_hat - y))

        w -= lr * grad_w
        b -= lr * grad_b

        history["loss"].append(loss)
        history["w"].append(w)
        history["b"].append(b)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Perceptron Training (for binary classification)
# ─────────────────────────────────────────────────────────────────────────────

def perceptron_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 50,
    activation: str = "sigmoid",
    loss: str = "bce",
    random_state: int = 42
) -> Tuple[np.ndarray, float, Dict]:
    """
    Train a single perceptron (logistic neuron) using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n, d)
    y : np.ndarray
        Binary labels in {0, 1}, shape (n,)
    lr : float
        Learning rate
    epochs : int
        Number of epochs
    activation : str
        Activation function name
    loss : str
        'bce' (binary cross-entropy) or 'mse'
    random_state : int
        Seed for reproducibility

    Returns
    -------
    w : np.ndarray
        Trained weights, shape (d,)
    b : float
        Trained bias
    history : dict
        'loss', 'accuracy' per epoch
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    w = rng.standard_normal(d) * 0.01
    b = 0.0

    history: Dict[str, List] = {"loss": [], "accuracy": []}

    for _ in range(epochs):
        # Forward
        Z = X @ w + b
        A = np.where(Z >= 0, 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500))),
                     np.exp(np.clip(Z, -500, 500)) / (1 + np.exp(np.clip(Z, -500, 500))))

        # Loss
        if loss == "bce":
            eps = 1e-12
            ep = float(-np.mean(y * np.log(A + eps) + (1 - y) * np.log(1 - A + eps)))
            dA = -(y / (A + eps) - (1 - y) / (1 - A + eps)) / n
        else:
            ep = float(np.mean(0.5 * (A - y) ** 2))
            dA = (A - y) / n

        # Backward (sigmoid derivative absorbed into dA * A*(1-A))
        dZ = dA * A * (1 - A)
        dw = X.T @ dZ
        db = float(np.sum(dZ))

        # Update
        w -= lr * dw
        b -= lr * db

        acc = float(np.mean((A >= 0.5).astype(float) == y))
        history["loss"].append(ep)
        history["accuracy"].append(acc)

    return w, b, history


# ─────────────────────────────────────────────────────────────────────────────
# ReLU Neuron Piecewise Analysis
# ─────────────────────────────────────────────────────────────────────────────

def relu_piecewise_boundary(
    w: np.ndarray,
    b: float
) -> Dict:
    """
    Analyse the switch-on / switch-off boundary for a single ReLU neuron.

    This implements the analysis from Practice Problem 2 in the slides.

    For a 2-D input: boundary is w[0]*x1 + w[1]*x2 + b = 0

    Returns
    -------
    dict with 'boundary_eq' string, 'slope', 'intercept' (if w[1]≠0)
    """
    result: Dict = {
        "boundary_eq": f"{w[0]:.3f}*x1 + {w[1]:.3f}*x2 + {b:.3f} = 0",
        "description": "Neuron is ON (a = z) when z > 0, OFF (a = 0) when z ≤ 0",
    }

    if len(w) == 2 and abs(w[1]) > 1e-10:
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        result["slope"] = slope
        result["intercept"] = intercept
        result["x2_formula"] = f"x2 = {slope:.3f}*x1 + {intercept:.3f}"

    return result


if __name__ == "__main__":
    # Quick smoke-test
    w = np.array([0.5, 0.2])
    b = 0.1
    x = np.array([1.0, 2.0])

    z, a = perceptron_forward(x, w, b, activation="sigmoid")
    print(f"z = {z:.4f}, a (sigmoid) = {a:.4f}")

    step = single_neuron_gradient_step(x=2.0, y=5.0, w=1.0, b=0.0, lr=0.1)
    print(f"Slide P3 step: w_new={step['w_new']:.1f}, b_new={step['b_new']:.1f}, "
          f"y_hat_new={step['y_hat_new']:.1f}")

    print("perceptron.py loaded OK ✓")
