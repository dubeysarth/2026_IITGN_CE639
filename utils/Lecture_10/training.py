"""
Training utilities for Lecture 10: Feedforward Neural Networks.

Covers loss functions, a full NumPy training loop for NumpyFNN, PyTorch wrappers,
and diagnostic plotting of training curves and decision boundaries.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
from .activations import safe_log


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error: L = (1/n) Σ (ŷ - y)²

    The standard regression loss. Penalises large errors quadratically.

    Parameters
    ----------
    y_true : np.ndarray, shape (n,) or (n, 1)
    y_pred : np.ndarray, shape (n,) or (n, 1)

    Returns
    -------
    float : scalar loss value
    """
    return float(np.mean((y_pred.ravel() - y_true.ravel()) ** 2))


def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE with respect to predictions: ∂L/∂ŷ = 2(ŷ - y) / n
    """
    n = len(y_true.ravel())
    return 2.0 * (y_pred.ravel() - y_true.ravel()) / n


def binary_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy: L = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]

    Used with sigmoid output for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels in {0, 1}
    y_pred : np.ndarray
        Predicted probabilities in (0, 1)
    """
    y = y_true.ravel()
    yh = np.clip(y_pred.ravel(), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(yh) + (1 - y) * np.log(1 - yh)))


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Multi-class Cross-Entropy: L = -(1/n) Σ_k y_k · log(ŷ_k)

    Used with softmax output for multi-class classification.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded labels, shape (n, n_classes)
    y_pred : np.ndarray
        Predicted probabilities (from softmax), same shape
    """
    yh = np.clip(y_pred, 1e-12, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(yh), axis=1)))


# ─────────────────────────────────────────────────────────────────────────────
# NumPy FNN Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_numpy_fnn(
    model,           # NumpyFNN instance
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    lr: float = 0.01,
    epochs: int = 100,
    batch_size: int = 32,
    loss_fn: str = "mse",
    verbose: int = 10,
    l2_lambda: float = 0.0,
    clip_grad: Optional[float] = None,
) -> Dict[str, List]:
    """
    Train a NumpyFNN using mini-batch gradient descent with backpropagation.

    This implements the full 4-step training loop from the slides:
        1. Forward pass → predictions
        2. Compute loss
        3. Backward pass → gradients (backprop)
        4. Update weights

    Parameters
    ----------
    model : NumpyFNN
        Initialised network from network.py
    X_train : np.ndarray, shape (n, d)
    y_train : np.ndarray, shape (n,) or (n, output_dim)
    X_val, y_val : optional validation data
    lr : float
        Learning rate η
    epochs : int
        Number of complete passes over training data
    batch_size : int
        Mini-batch size (None = full-batch GD)
    loss_fn : str
        'mse' or 'bce' (binary cross-entropy)
    verbose : int
        Print loss every `verbose` epochs (0 = silent)
    l2_lambda : float
        L2 regularisation coefficient λ
    clip_grad : float, optional
        Gradient clipping threshold (None = no clipping)

    Returns
    -------
    history : dict
        'train_loss', 'val_loss' (if val provided) lists per epoch
    """
    from .activations import get_derivative

    n = X_train.shape[0]
    y_train_2d = y_train.reshape(n, -1)
    history: Dict[str, List] = {"train_loss": [], "val_loss": []}

    if batch_size is None:
        batch_size = n

    for epoch in range(epochs):
        # ── Shuffle ──────────────────────────────────────────────────────
        idx = np.random.permutation(n)
        X_shuf = X_train[idx]
        y_shuf = y_train_2d[idx]

        # ── Mini-batch loop ───────────────────────────────────────────────
        batch_losses = []
        for start in range(0, n, batch_size):
            Xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]
            nb = Xb.shape[0]

            # ── 1. Forward pass ───────────────────────────────────────────
            _ = model.forward(Xb)               # populates model.cache
            y_hat = model.cache[f"a{model.n_layers}"]

            # ── 2. Loss ───────────────────────────────────────────────────
            if loss_fn == "mse":
                loss = mse_loss(yb, y_hat)
                # ∂L/∂a^[L]
                dA = 2.0 * (y_hat - yb) / nb
            elif loss_fn == "bce":
                loss = binary_cross_entropy_loss(yb, y_hat)
                eps = 1e-12
                yb_cl = np.clip(yb, 0, 1)
                yhat_cl = np.clip(y_hat, eps, 1 - eps)
                dA = (-(yb_cl / yhat_cl) + (1 - yb_cl) / (1 - yhat_cl)) / nb
            else:
                raise ValueError(f"Unknown loss_fn '{loss_fn}'; use 'mse' or 'bce'.")

            batch_losses.append(loss)

            # ── 3. Backward pass ──────────────────────────────────────────
            # Backprop through layers L → 1
            dW_list = [None] * model.n_layers
            db_list = [None] * model.n_layers

            dA_curr = dA

            for l in reversed(range(model.n_layers)):
                Z = model.cache[f"z{l+1}"]
                A_prev = model.cache[f"a{l}"] if l > 0 else model.cache["a0"]

                # Apply activation derivative
                deriv_fn = get_derivative(model.activation_names[l])
                dZ = dA_curr * deriv_fn(Z)           # (nb, n_out)

                # Gradients for W and b
                dW = dZ.T @ A_prev                   # (n_out, n_in)
                db = np.sum(dZ, axis=0)              # (n_out,)

                # L2 regularisation gradient
                if l2_lambda > 0:
                    dW += l2_lambda * model.W[l]

                # Gradient clipping
                if clip_grad is not None:
                    norm = np.linalg.norm(dW)
                    if norm > clip_grad:
                        dW *= clip_grad / norm

                dW_list[l] = dW
                db_list[l] = db

                # Propagate gradient to previous layer
                dA_curr = dZ @ model.W[l]            # (nb, n_in)

            # ── 4. Weight update ──────────────────────────────────────────
            for l in range(model.n_layers):
                model.W[l] -= lr * dW_list[l]
                model.b[l] -= lr * db_list[l]

        # ── Record epoch loss ─────────────────────────────────────────────
        epoch_loss = float(np.mean(batch_losses))
        history["train_loss"].append(epoch_loss)

        if X_val is not None and y_val is not None:
            y_val_hat = model.predict(X_val)
            if loss_fn == "mse":
                val_loss = mse_loss(y_val, y_val_hat)
            else:
                val_loss = binary_cross_entropy_loss(y_val, y_val_hat)
            history["val_loss"].append(val_loss)

        if verbose > 0 and (epoch + 1) % verbose == 0:
            val_str = (f", val_loss={history['val_loss'][-1]:.4f}"
                       if history["val_loss"] else "")
            print(f"Epoch {epoch+1:4d}/{epochs}: train_loss={epoch_loss:.4f}{val_str}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Training Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def train_pytorch_fnn(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs: int = 20,
    device: str = "cpu",
    val_loader=None,
    verbose: int = 5,
) -> Dict[str, List]:
    """
    Train a PyTorch FNN model for a given number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
    train_loader : torch.utils.data.DataLoader
    criterion : torch loss function
    optimizer : torch optimizer
    epochs : int
    device : str
        'cpu' or 'cuda'
    val_loader : DataLoader, optional
    verbose : int
        Print every N epochs

    Returns
    -------
    history : dict
        'train_loss', 'val_loss', 'train_acc', 'val_acc' per epoch
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    model.to(device)
    history: Dict[str, List] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * Xb.size(0)
            if out.ndim > 1 and out.shape[1] > 1:
                pred = out.argmax(dim=1)
                if yb.ndim > 1:
                    yb_cls = yb.argmax(dim=1)
                else:
                    yb_cls = yb
            else:
                pred = (out.squeeze() >= 0.5).long()
                yb_cls = yb.long()
            correct += (pred == yb_cls).sum().item()
            total += Xb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ── Validation ────────────────────────────────────────────────────
        if val_loader is not None:
            val_loss, val_acc = evaluate_pytorch(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if verbose > 0 and (epoch + 1) % verbose == 0:
            val_str = (f", val_loss={history['val_loss'][-1]:.4f}, "
                       f"val_acc={history['val_acc'][-1]:.3f}"
                       if val_loader else "")
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}{val_str}")

    return history


def evaluate_pytorch(
    model,
    loader,
    criterion,
    device: str = "cpu"
) -> Tuple[float, float]:
    """
    Evaluate a PyTorch model on a data loader.

    Returns
    -------
    (avg_loss, accuracy)
    """
    import torch

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * Xb.size(0)

            if out.ndim > 1 and out.shape[1] > 1:
                pred = out.argmax(dim=1)
                yb_cls = yb.argmax(dim=1) if yb.ndim > 1 else yb
            else:
                pred = (out.squeeze() >= 0.5).long()
                yb_cls = yb.long()

            correct += (pred == yb_cls).sum().item()
            total += Xb.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: Dict[str, List],
    title: str = "Training History",
    figsize: Tuple[int, int] = (13, 5)
) -> Tuple:
    """
    Plot training (and optional validation) loss and accuracy curves.

    Parameters
    ----------
    history : dict
        Keys: 'train_loss' (required), 'val_loss', 'train_acc', 'val_acc' (optional)
    title : str
    figsize : Tuple[int, int]

    Returns
    -------
    (fig, axes)
    """
    has_acc = "train_acc" in history and len(history["train_acc"]) > 0
    ncols = 2 if has_acc else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", markersize=3, linewidth=2,
            label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "r-s", markersize=3, linewidth=2,
                label="Val Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{title} — Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    if has_acc:
        ax = axes[1]
        ax.plot(epochs, history["train_acc"], "b-o", markersize=3, linewidth=2,
                label="Train Acc")
        if "val_acc" in history and history["val_acc"]:
            ax.plot(epochs, history["val_acc"], "r-s", markersize=3, linewidth=2,
                    label="Val Acc")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{title} — Accuracy", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_decision_regions(
    model_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 300,
    alpha: float = 0.4,
    figsize: Tuple[int, int] = (8, 7),
    title: str = "Decision Regions"
) -> Tuple:
    """
    Plot 2-D decision boundary and data points.

    Parameters
    ----------
    model_fn : Callable
        Function that takes X (n, 2) and returns predictions (n,) in {0, 1, ...}
    X : np.ndarray, shape (n, 2)
    y : np.ndarray, shape (n,)
    resolution : int
        Grid resolution
    alpha : float
        Background transparency
    figsize, title : standard matplotlib args

    Returns
    -------
    (fig, ax)
    """
    # Build grid
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    Z = model_fn(grid)
    if Z.ndim == 2:
        Z = Z[:, -1] if Z.shape[1] == 1 else np.argmax(Z, axis=1)
    Z = Z.reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx1, xx2, Z, alpha=alpha, cmap="RdYlBu")
    ax.contour(xx1, xx2, Z, colors="k", linewidths=1, linestyles="--")

    classes = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(classes)))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]],
                   marker=markers[i % len(markers)], s=50,
                   edgecolors="k", linewidths=0.5, label=f"Class {int(cls)}")

    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    return fig, ax


if __name__ == "__main__":
    # Smoke-test
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    print(f"MSE: {mse_loss(y_true, y_pred):.4f}")
    print(f"BCE: {binary_cross_entropy_loss(np.array([1,0,1]), np.array([0.9,0.1,0.8])):.4f}")
    print("training.py loaded OK ✓")
