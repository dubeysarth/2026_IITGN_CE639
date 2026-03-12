"""
Activation functions and their derivatives for Lecture 10: Feedforward Neural Networks.

Provides vectorised NumPy implementations of all activation functions discussed in
the slides, along with their first derivatives and a rich summary table.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
from typing import Dict, List, Tuple, Callable


# ─────────────────────────────────────────────────────────────────────────────
# Core Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation: σ(z) = 1 / (1 + e^{-z})

    Range: (0, 1)
    Squashes all real inputs to a probability-like output.
    Suffers from vanishing gradients for |z| >> 0.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values (any shape)

    Returns
    -------
    np.ndarray
        Sigmoid-activated values, same shape as z
    """
    # Numerically stable: avoid overflow for large positive z
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid: σ'(z) = σ(z) · (1 - σ(z))

    Maximum value is 0.25 at z = 0 — this is the root of the vanishing
    gradient problem: gradients get multiplied by ≤ 0.25 at each sigmoid layer.
    """
    s = sigmoid(z)
    return s * (1.0 - s)


# ─────────────────────────────────────────────────────────────────────────────

def tanh_act(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent: tanh(z) = (e^z - e^{-z}) / (e^z + e^{-z})

    Range: (-1, 1)
    Zero-centred unlike sigmoid — preferred for hidden layers historically.
    Still saturates for large |z|.
    """
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh: tanh'(z) = 1 - tanh²(z)

    Maximum value is 1.0 at z = 0.
    """
    return 1.0 - np.tanh(z) ** 2


# ─────────────────────────────────────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: ReLU(z) = max(0, z)

    Range: [0, ∞)
    Does not saturate for z > 0 → no vanishing gradient in the positive regime.
    Default activation for hidden layers in modern deep learning.
    Dead neurons problem: neurons with z ≤ 0 output 0 and receive zero gradient.
    """
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Sub-derivative of ReLU:
        ReLU'(z) = 1 if z > 0, else 0

    Note: technically undefined at z=0; we follow the convention of 0 there.
    """
    return (z > 0).astype(float)


# ─────────────────────────────────────────────────────────────────────────────

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU: max(α·z, z)

    Fixes the dead-neuron problem by allowing a small, non-zero gradient
    for negative inputs.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values
    alpha : float
        Slope for z < 0 (default 0.01)
    """
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU:
        1 if z > 0, else α
    """
    return np.where(z > 0, 1.0, alpha)


# ─────────────────────────────────────────────────────────────────────────────

def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit:
        z           if z > 0
        α(e^z - 1)  if z ≤ 0

    Smooth for negative inputs → avoids dead neurons while being differentiable.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values
    alpha : float
        Scale for negative part (default 1.0)
    """
    return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))


def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivative of ELU:
        1            if z > 0
        α·e^z        if z ≤ 0
    """
    return np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))


# ─────────────────────────────────────────────────────────────────────────────

def gelu(z: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (approximate):
        GELU(z) ≈ 0.5 · z · (1 + tanh(√(2/π) · (z + 0.044715 · z³)))

    Used in GPT, BERT, and modern transformers.
    Smooth approximation to ReLU with a slightly negative region for z ≈ -1.

    References
    ----------
    Hendrycks & Gimpel (2016). Gaussian Error Linear Units (GELUs).
    """
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))


def gelu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Approximate derivative of GELU (numerical, via finite differences).

    For educational purposes; in deep-learning frameworks this is computed
    via automatic differentiation.
    """
    h = 1e-5
    return (gelu(z + h) - gelu(z - h)) / (2 * h)


# ─────────────────────────────────────────────────────────────────────────────

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax activation for multi-class output layers.

        softmax(z)_k = e^{z_k} / Σ_j e^{z_j}

    Numerically stable: subtract max before exponentiation.

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values. For a single sample: shape (n_classes,).
        For batched: shape (batch, n_classes).
    axis : int
        Axis along which to apply softmax.

    Returns
    -------
    np.ndarray
        Probability distribution summing to 1 along `axis`.
    """
    # Shift for numerical stability
    z_shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Activation Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def get_activation(name: str) -> Callable:
    """
    Return an activation function by name string.

    Parameters
    ----------
    name : str
        One of: 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'gelu',
                'softmax', 'linear'

    Returns
    -------
    Callable
        The corresponding activation function (vectorised, NumPy).
    """
    registry = {
        "sigmoid": sigmoid,
        "tanh": tanh_act,
        "relu": relu,
        "leaky_relu": leaky_relu,
        "elu": elu,
        "gelu": gelu,
        "softmax": softmax,
        "linear": lambda z: z,
    }
    if name not in registry:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Available: {list(registry.keys())}")
    return registry[name]


def get_derivative(name: str) -> Callable:
    """
    Return the derivative of an activation function by name.

    Parameters
    ----------
    name : str
        Activation name (see get_activation).

    Returns
    -------
    Callable
    """
    registry = {
        "sigmoid": sigmoid_derivative,
        "tanh": tanh_derivative,
        "relu": relu_derivative,
        "leaky_relu": leaky_relu_derivative,
        "elu": elu_derivative,
        "gelu": gelu_derivative,
        "linear": lambda z: np.ones_like(z),
    }
    if name not in registry:
        raise ValueError(f"No derivative registered for '{name}'.")
    return registry[name]


def activation_summary_table() -> Dict[str, Dict]:
    """
    Return a dictionary of all supported activations with metadata.

    Each entry has:
        fn        : the activation function
        deriv     : its derivative function
        range     : output range as a string
        pros      : list of advantages
        cons      : list of disadvantages
        use_for   : typical use case

    Returns
    -------
    dict: {name → metadata_dict}
    """
    table = {
        "sigmoid": {
            "fn": sigmoid,
            "deriv": sigmoid_derivative,
            "formula": "1 / (1 + e^{-z})",
            "range": "(0, 1)",
            "pros": ["Output interpretable as probability", "Smooth & differentiable"],
            "cons": ["Vanishing gradients for |z| >> 0", "Not zero-centred"],
            "use_for": "Output layer for binary classification",
        },
        "tanh": {
            "fn": tanh_act,
            "deriv": tanh_derivative,
            "formula": "(e^z - e^{-z}) / (e^z + e^{-z})",
            "range": "(-1, 1)",
            "pros": ["Zero-centred", "Smooth & differentiable"],
            "cons": ["Still saturates for large |z|"],
            "use_for": "Hidden layers (RNNs, older architectures)",
        },
        "relu": {
            "fn": relu,
            "deriv": relu_derivative,
            "formula": "max(0, z)",
            "range": "[0, ∞)",
            "pros": ["No vanishing gradient for z>0", "Sparse activations", "Fast to compute"],
            "cons": ["Dead neurons: z ≤ 0 → zero gradient forever"],
            "use_for": "Hidden layers — modern default",
        },
        "leaky_relu": {
            "fn": leaky_relu,
            "deriv": leaky_relu_derivative,
            "formula": "max(αz, z), α=0.01",
            "range": "(-∞, ∞)",
            "pros": ["Fixes dead-neuron problem"],
            "cons": ["α is a hyperparameter"],
            "use_for": "Hidden layers when ReLU dead neurons are a problem",
        },
        "elu": {
            "fn": elu,
            "deriv": elu_derivative,
            "formula": "z if z>0 else α(e^z - 1)",
            "range": "(-α, ∞)",
            "pros": ["Smooth everywhere", "Negative outputs → near-zero mean activations"],
            "cons": ["Slower than ReLU (exp computation)"],
            "use_for": "Deep residual networks",
        },
        "gelu": {
            "fn": gelu,
            "deriv": gelu_derivative,
            "formula": "0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))",
            "range": "≈ (-0.17, ∞)",
            "pros": ["Smooth", "Slightly negative for z ≈ -1 (implicit regularisation)"],
            "cons": ["More expensive to compute"],
            "use_for": "Transformers (BERT, GPT)",
        },
        "linear": {
            "fn": lambda z: z,
            "deriv": lambda z: np.ones_like(z),
            "formula": "z",
            "range": "(-∞, ∞)",
            "pros": ["No saturation"],
            "cons": ["No nonlinearity: stacking gives a single linear layer"],
            "use_for": "Output layer for regression",
        },
    }
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Saturation & Dead-Neuron Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_saturation_fraction(
    z: np.ndarray,
    activation: str,
    threshold: float = 0.05
) -> float:
    """
    Compute the fraction of neurons that are saturated (near-zero gradient).

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values
    activation : str
        Activation name
    threshold : float
        Gradient magnitude below which a neuron is considered saturated

    Returns
    -------
    float
        Fraction of neurons with |gradient| < threshold (0 = none, 1 = all)
    """
    deriv_fn = get_derivative(activation)
    grads = np.abs(deriv_fn(z))
    return float(np.mean(grads < threshold))


def compute_dead_relu_fraction(z: np.ndarray) -> float:
    """
    Return the fraction of ReLU neurons that are permanently dead (z ≤ 0).

    Parameters
    ----------
    z : np.ndarray
        Pre-activation values

    Returns
    -------
    float
        Fraction of dead neurons
    """
    return float(np.mean(z <= 0))


# ─────────────────────────────────────────────────────────────────────────────
# Numerical Stability Utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically stable log — clips x to [eps, ∞)."""
    return np.log(np.clip(x, eps, None))


if __name__ == "__main__":
    # Quick smoke-test
    z = np.linspace(-4, 4, 9)
    print("z:", z.round(2))
    print("sigmoid:", sigmoid(z).round(3))
    print("tanh:   ", tanh_act(z).round(3))
    print("relu:   ", relu(z).round(3))
    print("gelu:   ", gelu(z).round(3))

    s = softmax(np.array([1.0, 2.0, 3.0]))
    print(f"softmax([1,2,3]) = {s.round(4)}, sum={s.sum():.6f}")

    table = activation_summary_table()
    print(f"\nActivation summary: {list(table.keys())}")
    print("activations.py loaded OK ✓")
