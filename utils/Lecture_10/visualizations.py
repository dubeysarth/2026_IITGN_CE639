"""
Rich visualisation utilities for Lecture 10: Feedforward Neural Networks.

Provides animated and static plots for activations, network diagrams,
forward-pass animations, gradient flow, initialisation effects, and
regularisation comparisons.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Activation Function Gallery
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_gallery(
    z_range: Tuple[float, float] = (-4.0, 4.0),
    figsize: Tuple[int, int] = (16, 10)
) -> Tuple:
    """
    Plot all activation functions and their derivatives side-by-side.

    Shows: sigmoid, tanh, relu, leaky_relu, elu, gelu in a 2×3 grid.
    Each panel shows function (blue) + derivative (red dashed).

    Returns
    -------
    fig, axes
    """
    from .activations import (sigmoid, sigmoid_derivative,
                               tanh_act, tanh_derivative,
                               relu, relu_derivative,
                               leaky_relu, leaky_relu_derivative,
                               elu, elu_derivative,
                               gelu, gelu_derivative)

    z = np.linspace(z_range[0], z_range[1], 500)

    activations_info = [
        ("Sigmoid", sigmoid(z), sigmoid_derivative(z), "(0, 1)", "Binary classif. output"),
        ("Tanh",    tanh_act(z), tanh_derivative(z), "(-1, 1)", "Zero-centred, RNNs"),
        ("ReLU",    relu(z), relu_derivative(z), "[0, ∞)", "Default hidden layer"),
        ("Leaky ReLU (α=0.01)", leaky_relu(z), leaky_relu_derivative(z), "(-∞, ∞)", "Fixes dead neurons"),
        ("ELU (α=1)",  elu(z), elu_derivative(z), "(-1, ∞)", "Smooth, near-zero mean"),
        ("GELU",   gelu(z), gelu_derivative(z), "≈(-0.17, ∞)", "Transformers"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, (name, fn, deriv, rng_str, use_for) in enumerate(activations_info):
        ax = axes[i]
        ax.plot(z, fn,    color="steelblue", linewidth=2.5, label=f"{name}")
        ax.plot(z, deriv, color="tomato",    linewidth=1.8, linestyle="--", label="Derivative")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_xlim(z_range)
        ax.set_xlabel("z", fontsize=11)
        ax.set_title(f"{name}\nRange: {rng_str}  |  Use: {use_for}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle("Activation Functions & Their Derivatives",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Network Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def plot_network_diagram(
    layer_sizes: List[int],
    activations: Optional[List[str]] = None,
    title: str = "Feedforward Neural Network",
    figsize: Tuple[int, int] = (14, 7),
    max_nodes_shown: int = 6,
) -> Tuple:
    """
    Draw a node-and-edge diagram of an FNN architecture.

    Layers are columns; neurons are circles; connections are edges.
    For large layers, shows a subset with '...' indicator.

    Parameters
    ----------
    layer_sizes : List[int]
    activations : List[str], optional
        Shown as labels between layers
    max_nodes_shown : int
        Maximum nodes to draw per layer (caps large layers visually)

    Returns
    -------
    fig, ax
    """
    n_layers = len(layer_sizes)
    if activations is None:
        activations = [""] * (n_layers - 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    layer_colors = ["#4CAF50", "#2196F3", "#FF9800"]  # input, hidden, output
    node_radius = 0.035

    # Positions
    node_positions = []
    for l, n in enumerate(layer_sizes):
        n_show = min(n, max_nodes_shown)
        ys = np.linspace(-(n_show - 1) / (2 * max_nodes_shown),
                          (n_show - 1) / (2 * max_nodes_shown), n_show)
        positions_l = [(l / (n_layers - 1), y) for y in ys]
        node_positions.append((positions_l, n > max_nodes_shown))

    # Draw edges
    for l in range(n_layers - 1):
        pos_l, _ = node_positions[l]
        pos_r, _ = node_positions[l + 1]
        for (x1, y1) in pos_l[:min(len(pos_l), max_nodes_shown)]:
            for (x2, y2) in pos_r[:min(len(pos_r), max_nodes_shown)]:
                ax.plot([x1, x2], [y1, y2], "gray", alpha=0.12, linewidth=0.6, zorder=1)

    # Draw nodes
    layer_labels = ["Input\nLayer"] + [f"Hidden {i}" for i in range(1, n_layers - 1)] + ["Output\nLayer"]
    for l, (positions, truncated) in enumerate(node_positions):
        color = layer_colors[0] if l == 0 else (layer_colors[2] if l == n_layers - 1 else layer_colors[1])
        for (x, y) in positions:
            circle = Circle((x, y), node_radius, color=color, ec="white", linewidth=1.5, zorder=3)
            ax.add_patch(circle)

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        # Ellipsis for truncated layers
        if truncated:
            ax.text(np.mean(xs), min(ys) - 0.08, "⋮",
                    ha="center", va="center", fontsize=16, color=color)

        # Layer label + size
        ax.text(positions[0][0], max(ys) + 0.12,
                f"{layer_labels[l]}\n({layer_sizes[l]})",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)

    # Activation labels between layers
    for l, act_name in enumerate(activations):
        if act_name:
            x_mid = (l / (n_layers - 1) + (l + 1) / (n_layers - 1)) / 2
            ax.text(x_mid, -0.75, act_name, ha="center", va="center",
                    fontsize=8, style="italic", color="gray",
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="gray", alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Flow Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_flow(
    gradients_per_layer: Dict[str, np.ndarray],
    title: str = "Gradient Magnitudes per Layer",
    figsize: Tuple[int, int] = (10, 5)
) -> Tuple:
    """
    Bar chart of average absolute gradient per layer.

    Used to visualise vanishing / exploding gradient problems.

    Parameters
    ----------
    gradients_per_layer : dict
        {layer_name → np.ndarray of gradient values}
    """
    names = list(gradients_per_layer.keys())
    means = [np.mean(np.abs(g)) for g in gradients_per_layer.values()]

    fig, ax = plt.subplots(figsize=figsize)

    # Colour-code: green = healthy, yellow = small, red = vanishing
    colors = []
    for m in means:
        if m > 1e-1:
            colors.append("tomato")    # exploding
        elif m > 1e-3:
            colors.append("steelblue") # healthy
        elif m > 1e-6:
            colors.append("gold")      # small
        else:
            colors.append("gray")      # vanishing

    bars = ax.bar(range(len(names)), means, color=colors, edgecolor="black", linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("|Gradient| (mean)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{val:.2e}", ha="center", va="bottom", fontsize=8, rotation=30)

    # Legend
    legend_items = [
        mpatches.Patch(color="steelblue", label="Healthy (>1e-3)"),
        mpatches.Patch(color="gold", label="Small (1e-6 – 1e-3)"),
        mpatches.Patch(color="gray", label="Vanishing (<1e-6)"),
        mpatches.Patch(color="tomato", label="Exploding (>0.1)"),
    ]
    ax.legend(handles=legend_items, fontsize=9, loc="upper right")
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_init_comparison(
    layer_sizes: List[int],
    init_methods: Optional[List[str]] = None,
    n_samples: int = 1000,
    figsize: Tuple[int, int] = (16, 8)
) -> Tuple:
    """
    Compare activation distributions after different weight initialisations.

    Passes random input through the network and plots histograms of
    activations at each hidden layer for each init method.

    Parameters
    ----------
    layer_sizes : List[int]
    init_methods : List[str]
    n_samples : int
        Number of random input samples
    """
    from .network import NumpyFNN

    if init_methods is None:
        init_methods = ["zero", "random_small", "xavier", "he"]

    n_hidden = len(layer_sizes) - 2  # exclude input and output layers
    n_init = len(init_methods)

    fig, axes = plt.subplots(n_hidden, n_init, figsize=figsize)
    if n_hidden == 1:
        axes = axes[np.newaxis, :]

    X_test = np.random.randn(n_samples, layer_sizes[0])

    colors = plt.cm.Set2(np.linspace(0, 1, n_init))

    for j, method in enumerate(init_methods):
        try:
            model = NumpyFNN(layer_sizes,
                             activations=["relu"] * (len(layer_sizes) - 2) + ["linear"],
                             init_method=method)
            _ = model.forward(X_test)
            for i in range(n_hidden):
                a = model.cache[f"a{i+1}"].ravel()
                ax = axes[i, j]
                ax.hist(a, bins=40, color=colors[j], alpha=0.8, edgecolor="none")
                ax.set_title(f"{method}\nLayer {i+1}", fontsize=9, fontweight="bold")
                ax.set_xlabel("Activation", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                dead = float(np.mean(a == 0)) * 100
                ax.text(0.7, 0.92, f"{dead:.0f}% dead",
                        transform=ax.transAxes, fontsize=8, color="red",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
                ax.grid(True, alpha=0.2)
        except Exception as e:
            for i in range(n_hidden):
                axes[i, j].text(0.5, 0.5, f"Error:\n{str(e)[:40]}",
                                transform=axes[i, j].transAxes,
                                ha="center", va="center", fontsize=8, color="red")

    plt.suptitle(f"Activation Distributions After Different Initialisations\n"
                 f"Network: {layer_sizes}, ReLU activations, {n_samples} random inputs",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Regularisation Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_regularization_comparison(
    histories: Dict[str, Dict],
    figsize: Tuple[int, int] = (14, 5)
) -> Tuple:
    """
    Overlay multiple training histories for regularisation comparison.

    Parameters
    ----------
    histories : dict
        {label → {'train_loss': [...], 'val_loss': [...]}}

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(histories)))

    for (label, hist), col in zip(histories.items(), colors):
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], "-", color=col, linewidth=2, label=f"{label} (train)")
        if "val_loss" in hist and hist["val_loss"]:
            axes[0].plot(epochs, hist["val_loss"], "--", color=col, linewidth=1.5, label=f"{label} (val)")

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Generalisation gap (val - train at each epoch)
    for (label, hist), col in zip(histories.items(), colors):
        if "val_loss" in hist and hist["val_loss"]:
            gap = np.array(hist["val_loss"]) - np.array(hist["train_loss"][:len(hist["val_loss"])])
            axes[1].plot(np.arange(1, len(gap) + 1), gap, "-", color=col, linewidth=2, label=label)

    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Generalisation Gap (val - train)", fontsize=12)
    axes[1].set_title("Overfitting Indicator", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Loss Landscape (2-D slice)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_landscape_2d(
    loss_fn,
    param_ranges: Tuple[np.ndarray, np.ndarray],
    true_params: Optional[Tuple[float, float]] = None,
    title: str = "Loss Landscape (2D Slice)",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple:
    """
    Plot a 2-D slice of the loss surface.

    Parameters
    ----------
    loss_fn : Callable
        Function (w1, w2) → scalar loss value (vectorised over grids)
    param_ranges : Tuple[np.ndarray, np.ndarray]
        (w1_values, w2_values) for the meshgrid
    true_params : Tuple[float, float], optional
        Optimal (w1*, w2*) to mark with a star
    """
    w1_vals, w2_vals = param_ranges
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    L = np.vectorize(loss_fn)(W1, W2)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(W1, W2, L, levels=40, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="Loss")
    ax.contour(W1, W2, L, levels=20, colors="white", alpha=0.3, linewidths=0.5)

    if true_params is not None:
        ax.scatter(*true_params, c="yellow", s=300, marker="*", zorder=5,
                   edgecolors="black", linewidths=1.5, label="Optimum")
        ax.legend(fontsize=11)

    ax.set_xlabel("Parameter w₁", fontsize=12)
    ax.set_ylabel("Parameter w₂", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Depth vs Width Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_depth_vs_width(
    results: Dict[Tuple[int, int], float],
    depth_values: List[int],
    width_values: List[int],
    title: str = "Validation Loss: Depth × Width",
    figsize: Tuple[int, int] = (9, 7)
) -> Tuple:
    """
    Heatmap of validation loss over a grid of network depths and widths.

    Parameters
    ----------
    results : Dict[Tuple[int, int], float]
        {(depth, width) → val_loss}
    depth_values : List[int]
        Row labels (number of hidden layers)
    width_values : List[int]
        Column labels (neurons per layer)
    """
    matrix = np.array([[results.get((d, w), np.nan)
                         for w in width_values]
                        for d in depth_values])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="plasma_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Validation Loss")

    ax.set_xticks(range(len(width_values)))
    ax.set_yticks(range(len(depth_values)))
    ax.set_xticklabels(width_values)
    ax.set_yticklabels(depth_values)
    ax.set_xlabel("Width (neurons/layer)", fontsize=12)
    ax.set_ylabel("Depth (hidden layers)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(len(depth_values)):
        for j in range(len(width_values)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if val > matrix.mean() else "black")

    # Highlight minimum
    min_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
    ax.add_patch(plt.Rectangle((min_idx[1] - 0.4, min_idx[0] - 0.4), 0.8, 0.8,
                                fill=False, edgecolor="lime", linewidth=3))

    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Batch Normalization Effect
# ─────────────────────────────────────────────────────────────────────────────

def plot_batch_norm_effect(
    histories_with_bn: Dict,
    histories_without_bn: Dict,
    figsize: Tuple[int, int] = (14, 5)
) -> Tuple:
    """
    Compare training curves with and without BatchNorm.

    Parameters
    ----------
    histories_with_bn, histories_without_bn : dicts
        Each has 'train_loss' and optionally 'val_loss'
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs_bn = np.arange(1, len(histories_with_bn["train_loss"]) + 1)
    epochs_no = np.arange(1, len(histories_without_bn["train_loss"]) + 1)

    axes[0].plot(epochs_no, histories_without_bn["train_loss"], "r-", linewidth=2, label="No BN (train)")
    axes[0].plot(epochs_bn, histories_with_bn["train_loss"], "b-", linewidth=2, label="With BN (train)")
    if "val_loss" in histories_with_bn:
        axes[0].plot(epochs_bn, histories_with_bn["val_loss"], "b--", linewidth=1.5, label="With BN (val)")
    if "val_loss" in histories_without_bn:
        axes[0].plot(epochs_no, histories_without_bn["val_loss"], "r--", linewidth=1.5, label="No BN (val)")

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss: With vs Without Batch Norm", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Stability — rolling std of loss
    def rolling_std(x, w=5):
        return np.array([np.std(x[max(0, i-w):i+1]) for i in range(len(x))])

    std_bn = rolling_std(histories_with_bn["train_loss"])
    std_no = rolling_std(histories_without_bn["train_loss"])

    axes[1].plot(epochs_no, std_no, "r-", linewidth=2, label="No BN")
    axes[1].plot(epochs_bn, std_bn, "b-", linewidth=2, label="With BN")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Rolling Loss StdDev", fontsize=12)
    axes[1].set_title("Training Stability", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    fig, _ = plot_activation_gallery()
    plt.savefig("/tmp/act_gallery_test.png", dpi=80, bbox_inches="tight")
    plt.close()

    fig, _ = plot_network_diagram([6, 32, 16, 1], activations=["ReLU", "ReLU", "Linear"])
    plt.savefig("/tmp/net_diagram_test.png", dpi=80, bbox_inches="tight")
    plt.close()

    print("visualizations.py loaded OK ✓")
