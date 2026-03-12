"""
From-scratch feedforward neural network (NumPy) for Lecture 10.

Implements a full multi-layer FNN with configurable depth, width, initialisation
strategy, and forward-pass step-by-step generator (used for animation).

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
from typing import Dict, Generator, List, Optional, Tuple
from .activations import get_activation, get_derivative


# ─────────────────────────────────────────────────────────────────────────────
# Weight Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_weights(
    n_in: int,
    n_out: int,
    method: str = "he",
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise weight matrix and bias vector for a single layer.

    Parameters
    ----------
    n_in : int
        Number of input neurons (fan-in)
    n_out : int
        Number of output neurons
    method : str
        Initialisation strategy:
        - 'xavier' / 'glorot': σ = √(1/n_in)  (for sigmoid/tanh)
        - 'he' / 'he_normal': σ = √(2/n_in)   (for ReLU)
        - 'zero': all zeros (WRONG — breaks symmetry, for demonstration)
        - 'random_small': uniform [-0.1, 0.1]
        - 'ones': all ones (WRONG — for demonstration)
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns
    -------
    W : np.ndarray, shape (n_out, n_in)
        Weight matrix
    b : np.ndarray, shape (n_out,)
        Bias vector (always zero-initialised)

    Notes
    -----
    Biases are always initialised to zero; symmetry breaking only applies to W.
    With zero weights all neurons in a layer compute the same function.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    b = np.zeros(n_out)

    if method in ("xavier", "glorot"):
        std = np.sqrt(1.0 / n_in)
        W = rng.normal(0.0, std, size=(n_out, n_in))
    elif method in ("he", "he_normal"):
        std = np.sqrt(2.0 / n_in)
        W = rng.normal(0.0, std, size=(n_out, n_in))
    elif method == "zero":
        W = np.zeros((n_out, n_in))
    elif method == "random_small":
        W = rng.uniform(-0.1, 0.1, size=(n_out, n_in))
    elif method == "ones":
        W = np.ones((n_out, n_in))
    else:
        raise ValueError(f"Unknown init method '{method}'. "
                         f"Choose: xavier, glorot, he, he_normal, zero, random_small, ones")

    return W, b


# ─────────────────────────────────────────────────────────────────────────────
# NumpyFNN — From-Scratch Multi-Layer Network
# ─────────────────────────────────────────────────────────────────────────────

class NumpyFNN:
    """
    A fully-connected feedforward neural network implemented in pure NumPy.

    Useful for:
    - Understanding the forward pass step-by-step
    - Demonstrating initialisation effects
    - Animating gradient flow through layers

    Attributes
    ----------
    layer_sizes : list of int
        [n_input, n_hidden_1, ..., n_hidden_L, n_output]
    activations : list of str
        Activation name for each layer (length = n_layers)
    W : list of np.ndarray
        Weight matrices
    b : list of np.ndarray
        Bias vectors
    cache : dict
        Stores z and a for each layer after forward pass (needed for backprop)

    Notes
    -----
    This class deliberately exposes internal arrays for educational transparency.
    For production use, switch to `architectures.SimpleFNN` (PyTorch).
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[str]] = None,
        init_method: str = "he",
        random_state: int = 42
    ):
        """
        Initialise the network.

        Parameters
        ----------
        layer_sizes : List[int]
            Layer sizes including input and output.
            Example: [6, 32, 16, 1] → 2 hidden layers.
        activations : List[str], optional
            Activation for each layer (excluding input).
            Default: relu for hidden, linear for output.
        init_method : str
            Weight initialisation strategy.
        random_state : int
            Seed for reproducibility.
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # number of weight layers
        self.init_method = init_method

        # Default activations: relu for hidden, linear for output
        if activations is None:
            activations = ["relu"] * (self.n_layers - 1) + ["linear"]
        if len(activations) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} activations, got {len(activations)}")
        self.activation_names = activations

        # Initialise weights
        rng = np.random.default_rng(random_state)
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []

        for l in range(self.n_layers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l + 1]
            W_l, b_l = init_weights(n_in, n_out, init_method, rng)
            self.W.append(W_l)
            self.b.append(b_l)

        self.cache: Dict = {}

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass and cache intermediate values.

        Parameters
        ----------
        X : np.ndarray
            Input, shape (n_samples, n_features) or (n_features,)

        Returns
        -------
        np.ndarray
            Output activations a^[L], shape (n_samples, n_output)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.cache = {"a0": X}
        A = X  # a^[0] = x

        for l in range(self.n_layers):
            Z = A @ self.W[l].T + self.b[l]       # (n, n_out)
            act_fn = get_activation(self.activation_names[l])
            A = act_fn(Z)
            self.cache[f"z{l+1}"] = Z
            self.cache[f"a{l+1}"] = A

        return A

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forward pass without caching (for inference)."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        A = X
        for l in range(self.n_layers):
            Z = A @ self.W[l].T + self.b[l]
            A = get_activation(self.activation_names[l])(Z)
        return A

    # ── Dimensions ───────────────────────────────────────────────────────────

    def get_weight_dimensions(self) -> List[str]:
        """Return human-readable dimension string for each weight matrix."""
        dims = []
        for l in range(self.n_layers):
            dims.append(f"W[{l+1}]: ({self.layer_sizes[l+1]} × {self.layer_sizes[l]}), "
                        f"b[{l+1}]: ({self.layer_sizes[l+1]},)")
        return dims

    def summary(self) -> str:
        """Print a layer-by-layer summary."""
        lines = ["NumpyFNN Summary", "=" * 50]
        total_params = 0
        lines.append(f"{'Layer':<12} {'Shape (W)':<20} {'Activation':<12} {'Params':>10}")
        lines.append("-" * 55)
        for l in range(self.n_layers):
            n_in, n_out = self.layer_sizes[l], self.layer_sizes[l+1]
            p = n_out * n_in + n_out   # weights + biases
            total_params += p
            lines.append(f"Layer {l+1:<6} ({n_out}×{n_in}){'':<8} "
                         f"{self.activation_names[l]:<12} {p:>10,}")
        lines.append("=" * 55)
        lines.append(f"{'Total parameters':>44} {total_params:>10,}")
        return "\n".join(lines)

    # ── Activation statistics ─────────────────────────────────────────────────

    def get_activation_stats(self) -> Dict:
        """
        After calling forward(), return statistics of pre-activations per layer.

        Useful for diagnosing vanishing/exploding gradients and dead neurons.
        """
        if not self.cache:
            raise RuntimeError("Call forward() first to populate cache.")

        stats = {}
        for l in range(1, self.n_layers + 1):
            z = self.cache.get(f"z{l}")
            a = self.cache.get(f"a{l}")
            if z is not None:
                stats[f"layer_{l}"] = {
                    "z_mean": float(np.mean(z)),
                    "z_std": float(np.std(z)),
                    "a_mean": float(np.mean(a)),
                    "a_std": float(np.std(a)),
                    "dead_frac": float(np.mean(a == 0)),  # works for ReLU
                }
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Step-by-Step Forward Pass (for animation)
# ─────────────────────────────────────────────────────────────────────────────

def forward_pass_step_by_step(
    X: np.ndarray,
    model: NumpyFNN
) -> Generator[Dict, None, None]:
    """
    Generator that yields the state after each layer's forward computation.

    Used to construct animated forward-pass visualisations.

    Parameters
    ----------
    X : np.ndarray
        Single input sample, shape (n_features,) or (1, n_features)
    model : NumpyFNN
        Initialised network

    Yields
    ------
    dict with keys:
        'layer'       : int, layer index (1-indexed)
        'z'           : np.ndarray, pre-activation values
        'a'           : np.ndarray, post-activation values
        'activation'  : str, activation name
        'W'           : np.ndarray, weight matrix
        'b'           : np.ndarray, bias vector
        'prev_a'      : np.ndarray, previous layer activations
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    A = X
    for l in range(model.n_layers):
        Z = A @ model.W[l].T + model.b[l]
        act_fn = get_activation(model.activation_names[l])
        A_new = act_fn(Z)

        yield {
            "layer": l + 1,
            "z": Z.ravel(),
            "a": A_new.ravel(),
            "activation": model.activation_names[l],
            "W": model.W[l],
            "b": model.b[l],
            "prev_a": A.ravel(),
        }

        A = A_new


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Counting Utility
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(layer_sizes: List[int]) -> Dict:
    """
    Count total, trainable and per-layer parameters for an FNN.

    Parameters
    ----------
    layer_sizes : List[int]
        Layer sizes including input and output.

    Returns
    -------
    dict with 'total', 'per_layer' (list of dicts)
    """
    per_layer = []
    total = 0
    for l in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
        weights = n_out * n_in
        biases = n_out
        params = weights + biases
        total += params
        per_layer.append({
            "layer": l + 1,
            "n_in": n_in,
            "n_out": n_out,
            "weights": weights,
            "biases": biases,
            "params": params,
        })

    return {"total": total, "per_layer": per_layer}


# ─────────────────────────────────────────────────────────────────────────────
# Collapse Demo Utility
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_linear_collapse(layer_sizes: List[int]) -> Dict:
    """
    Demonstrate that a stack of linear layers (no activations) collapses
    to a single linear transformation.

    Parameters
    ----------
    layer_sizes : List[int]
        E.g., [2, 4, 4, 1] — 2 hidden layers

    Returns
    -------
    dict with:
        'W_product' : the product W^[L] @ ... @ W^[1]
        'rank'      : rank of the product matrix
        'message'   : explanation string
    """
    rng = np.random.default_rng(0)
    matrices = []

    for l in range(len(layer_sizes) - 1):
        n_in, n_out = layer_sizes[l], layer_sizes[l + 1]
        W = rng.normal(0, 0.5, (n_out, n_in))
        matrices.append(W)

    # Compute the product W^[L] @ ... @ W^[1]
    W_prod = matrices[-1]
    for W in reversed(matrices[:-1]):
        W_prod = W_prod @ W

    rank = np.linalg.matrix_rank(W_prod)
    max_rank = min(layer_sizes[0], layer_sizes[-1])

    return {
        "W_product": W_prod,
        "W_product_shape": W_prod.shape,
        "rank": rank,
        "max_possible_rank": max_rank,
        "message": (
            f"A {len(layer_sizes)-1}-layer linear network collapses to a single "
            f"{W_prod.shape[0]}×{W_prod.shape[1]} linear transformation.\n"
            f"Rank = {rank} (max possible = {max_rank}).\n"
            f"Depth buys NOTHING without nonlinear activations!"
        ),
    }


if __name__ == "__main__":
    # Smoke-test
    model = NumpyFNN([6, 32, 16, 1], activations=["relu", "relu", "linear"])
    print(model.summary())

    X = np.random.randn(5, 6)
    out = model.forward(X)
    print(f"Forward pass output shape: {out.shape}")

    stats = model.get_activation_stats()
    print(f"Layer stats: {list(stats.keys())}")

    collapse = demonstrate_linear_collapse([2, 8, 8, 1])
    print(collapse["message"])

    print("\nnetwork.py loaded OK ✓")
