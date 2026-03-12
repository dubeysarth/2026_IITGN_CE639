"""
Interactive ipywidgets for Lecture 10: Feedforward Neural Networks.

Provides live, interactive explorers for activations, forward pass,
network architecture, training dynamics, and weight initialisation.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def _require_ipywidgets():
    """Check for ipywidgets and return interact + widgets, or print instructions."""
    try:
        from ipywidgets import interact, widgets
        return interact, widgets
    except ImportError:
        print("⚠️ ipywidgets not found. Install with:\n    pip install ipywidgets")
        print("   Then enable with:\n    jupyter nbextension enable --py widgetsnbextension")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Widget 1 — Activation Function Explorer
# ─────────────────────────────────────────────────────────────────────────────

def activation_explorer_widget():
    """
    Interactive widget for exploring activation functions.

    Controls:
        • Activation type (dropdown)
        • Z range (slider)
        • Show derivative (checkbox)

    Shows the activation and optionally its derivative over the selected range.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    from .activations import get_activation, get_derivative, activation_summary_table

    act_names = ["sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "linear"]
    table = activation_summary_table()

    def update(activation="relu", z_range=4.0, show_derivative=True):
        z = np.linspace(-z_range, z_range, 500)
        fn = get_activation(activation)
        a = fn(z)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z, a, "steelblue", linewidth=2.5, label=f"{activation}(z)")

        if show_derivative and activation in table:
            deriv_fn = table[activation]["deriv"]
            d = deriv_fn(z)
            ax.plot(z, d, "tomato", linewidth=1.8, linestyle="--", label=f"{activation}'(z)")

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xlim(-z_range, z_range)
        ax.set_xlabel("z (pre-activation)", fontsize=12)
        ax.set_ylabel("a = g(z)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Info box
        if activation in table:
            info = table[activation]
            ax.set_title(
                f"{activation}   Range: {info['range']}   Use: {info['use_for']}",
                fontsize=11, fontweight="bold"
            )
        plt.tight_layout()
        plt.show()

    interact(
        update,
        activation=widgets.Dropdown(options=act_names, value="relu",
                                     description="Activation:"),
        z_range=widgets.FloatSlider(min=1.0, max=8.0, step=0.5, value=4.0,
                                     description="Z range:"),
        show_derivative=widgets.Checkbox(value=True, description="Show derivative")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Widget 2 — Forward Pass Explorer
# ─────────────────────────────────────────────────────────────────────────────

def forward_pass_widget():
    """
    Interactive widget for a single-layer forward pass.

    Allows manual input of x1, x2, weights W, bias b, and activation.
    Displays z = Wx + b and a = g(z) with a compact diagram.

    Great for understanding Practice Problem 1 from the slides.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    from .activations import get_activation

    def update(x1=1.0, x2=2.0,
               w11=1.0, w12=-1.0, w21=0.5, w22=2.0,
               b1=0.0, b2=-1.0,
               activation="relu"):

        x = np.array([x1, x2])
        W = np.array([[w11, w12], [w21, w22]])
        b = np.array([b1, b2])

        z = W @ x + b
        fn = get_activation(activation)
        a = fn(z)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # Input
        axes[0].bar(["x₁", "x₂"], [x1, x2], color=["#4CAF50", "#4CAF50"])
        axes[0].set_title("Input x", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Value", fontsize=11)
        for i, v in enumerate([x1, x2]):
            axes[0].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11)
        axes[0].grid(axis="y", alpha=0.3)

        # Pre-activation z
        axes[1].bar(["z₁", "z₂"], z, color=["#FF9800", "#FF9800"])
        axes[1].set_title(f"Pre-activation z = Wx + b", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("z", fontsize=11)
        for i, v in enumerate(z):
            axes[1].text(i, v + max(0, 0.05 * abs(v)), f"{v:.2f}", ha="center", fontsize=11)
        axes[1].grid(axis="y", alpha=0.3)

        # Post-activation a
        colors = ["#2196F3" if v > 0 else "#9E9E9E" for v in a]
        axes[2].bar(["a₁", "a₂"], a, color=colors)
        axes[2].set_title(f"Post-activation a = {activation}(z)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("a", fontsize=11)
        for i, v in enumerate(a):
            axes[2].text(i, v + max(0, 0.05 * abs(v)), f"{v:.3f}", ha="center", fontsize=11)
        axes[2].grid(axis="y", alpha=0.3)

        # Print detailed computation
        plt.suptitle(
            f"W = [[{w11},{w12}],[{w21},{w22}]], b = [{b1},{b2}]\n"
            f"z = [{z[0]:.3f}, {z[1]:.3f}],  a = [{a[0]:.3f}, {a[1]:.3f}]",
            fontsize=10, y=1.01
        )
        plt.tight_layout()
        plt.show()

    interact(
        update,
        x1=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=1.0, description="x₁:"),
        x2=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=2.0, description="x₂:"),
        w11=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=1.0, description="W[1,1]:"),
        w12=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=-1.0, description="W[1,2]:"),
        w21=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0.5, description="W[2,1]:"),
        w22=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=2.0, description="W[2,2]:"),
        b1=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=0.0, description="b₁:"),
        b2=widgets.FloatSlider(min=-3.0, max=3.0, step=0.1, value=-1.0, description="b₂:"),
        activation=widgets.Dropdown(options=["relu", "sigmoid", "tanh", "leaky_relu", "linear"],
                                    description="Activation:")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Widget 3 — Network Builder
# ─────────────────────────────────────────────────────────────────────────────

def network_builder_widget():
    """
    Interactive widget for exploring network architectures.

    Controls:
        • Input dimension
        • Number of hidden layers
        • Neurons per hidden layer
        • Activation function

    Displays parameter count and a simplified network diagram.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    from .network import count_parameters as count_np_params
    from .visualizations import plot_network_diagram

    def update(input_dim=6, n_hidden=2, neurons_per_layer=32,
               output_dim=1, activation="relu"):
        layer_sizes = [input_dim] + [neurons_per_layer] * n_hidden + [output_dim]
        params = count_np_params(layer_sizes)

        print(f"\n{'=' * 55}")
        print(f"  Network: {layer_sizes}")
        print(f"  Activation: {activation}")
        print(f"{'=' * 55}")
        print(f"  {'Layer':<15} {'Shape (W)':<20} {'Params':>8}")
        print(f"  {'-' * 45}")
        for p in params["per_layer"]:
            print(f"  Layer {p['layer']:<9} ({p['n_out']} × {p['n_in']}){'':<8} {p['params']:>8,}")
        print(f"  {'-' * 45}")
        print(f"  {'TOTAL':<36} {params['total']:>8,}")
        print(f"{'=' * 55}")

        acts = [activation] * n_hidden + ["linear"]
        fig, ax = plot_network_diagram(layer_sizes, activations=acts,
                                        title=f"FNN: {layer_sizes}  ({params['total']:,} params)")
        plt.show()

    interact(
        update,
        input_dim=widgets.IntSlider(min=1, max=20, value=6, description="Input dim:"),
        n_hidden=widgets.IntSlider(min=1, max=8, value=2, description="Hidden layers:"),
        neurons_per_layer=widgets.IntSlider(min=4, max=256, step=4, value=32,
                                            description="Neurons/layer:"),
        output_dim=widgets.IntSlider(min=1, max=10, value=1, description="Output dim:"),
        activation=widgets.Dropdown(options=["relu", "tanh", "sigmoid", "leaky_relu", "elu"],
                                    description="Activation:")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Widget 4 — Training Playground
# ─────────────────────────────────────────────────────────────────────────────

def training_playground_widget():
    """
    Interactive widget for exploring neural network training dynamics.

    Simulates training curves for different hyperparameter combinations.
    Controls: learning rate, batch size, epochs, regularisation type.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    def update(lr=0.01, batch_size=32, epochs=50, l2_lambda=0.0, dropout=0.0):
        np.random.seed(42)

        # Simulate meaningful training curves based on hyperparameters
        t = np.arange(epochs)

        # Base convergence speed tied to learning rate
        speed = np.clip(lr * 100, 0.3, 5.0)
        noise_scale = 0.05 + 0.02 * (32 / max(batch_size, 1))

        # Train loss
        train_loss = 1.2 * np.exp(-speed * t / epochs) + np.random.randn(epochs) * noise_scale * 0.5

        # Val loss: gap increases with overfitting (low l2/dropout)
        overfit_gap = max(0, 0.4 - 5 * l2_lambda - 3 * dropout)
        val_loss = (train_loss + overfit_gap * (1 - np.exp(-t / (0.4 * epochs)))
                    + np.random.randn(epochs) * noise_scale)

        # Accuracy (rough estimate)
        train_acc = 1 - train_loss / 1.5 + np.random.randn(epochs) * 0.02
        val_acc   = 1 - val_loss / 1.5 + np.random.randn(epochs) * 0.03

        train_acc = np.clip(train_acc, 0, 1)
        val_acc   = np.clip(val_acc,   0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Loss
        axes[0].plot(t + 1, train_loss, "b-", linewidth=2, label="Train Loss")
        axes[0].plot(t + 1, val_loss,   "r-", linewidth=2, label="Val Loss")
        axes[0].fill_between(t + 1, train_loss, val_loss, alpha=0.1, color="red",
                             label="Generalisation Gap")
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(f"Loss  |  LR={lr}, BS={batch_size}, λ={l2_lambda}, drop={dropout}",
                          fontsize=11, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.5)

        # Accuracy
        axes[1].plot(t + 1, train_acc, "b-", linewidth=2, label="Train Accuracy")
        axes[1].plot(t + 1, val_acc,   "r-", linewidth=2, label="Val Accuracy")
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title("Accuracy Curves", fontsize=11, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

        plt.tight_layout()
        plt.show()

        print(f"\n📊 Final results at epoch {epochs}:")
        print(f"   Train loss: {train_loss[-1]:.4f} | Val loss: {val_loss[-1]:.4f}")
        print(f"   Gap: {val_loss[-1] - train_loss[-1]:.4f} "
              f"({'overfitting' if (val_loss[-1] - train_loss[-1]) > 0.15 else 'OK'})")

    interact(
        update,
        lr=widgets.FloatLogSlider(min=-4, max=-1, step=0.25, value=0.01,
                                   description="LR:", readout_format=".4f"),
        batch_size=widgets.Dropdown(options=[8, 16, 32, 64, 128, 256],
                                    value=32, description="Batch size:"),
        epochs=widgets.IntSlider(min=10, max=200, step=10, value=50, description="Epochs:"),
        l2_lambda=widgets.FloatSlider(min=0.0, max=0.1, step=0.005, value=0.0,
                                      description="L2 λ:"),
        dropout=widgets.FloatSlider(min=0.0, max=0.5, step=0.05, value=0.0,
                                     description="Dropout:")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Widget 5 — Initialisation Explorer
# ─────────────────────────────────────────────────────────────────────────────

def initialization_widget():
    """
    Interactive widget comparing weight initialisations.

    Controls:
        • Init method (zero, random_small, xavier, he)
        • Network depth

    Displays activation histograms for each hidden layer.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    from .visualizations import plot_init_comparison

    def update(init_method="he", depth=3, width=64, input_dim=10):
        layer_sizes = [input_dim] + [width] * depth + [1]
        fig, _ = plot_init_comparison(
            layer_sizes, init_methods=[init_method, "xavier", "random_small", "zero"],
            n_samples=500
        )
        plt.show()

    interact(
        update,
        init_method=widgets.Dropdown(options=["he", "xavier", "random_small", "zero"],
                                      description="Init:"),
        depth=widgets.IntSlider(min=1, max=6, value=3, description="Depth:"),
        width=widgets.IntSlider(min=8, max=128, step=8, value=64, description="Width:"),
        input_dim=widgets.IntSlider(min=2, max=20, value=10, description="Input dim:")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Widget 6 — Learning Rate Effect
# ─────────────────────────────────────────────────────────────────────────────

def learning_rate_widget():
    """
    Demonstrate the effect of learning rate on gradient descent convergence.

    Animates GD steps on a 1-D quadratic loss landscape.
    Controls: learning rate slider.
    """
    interact, widgets = _require_ipywidgets()
    if interact is None:
        return

    def update(lr=0.1, n_steps=20):
        # Simple 1-D quadratic: L = (w - 3)^2
        def loss(w):
            return (w - 3.0) ** 2

        def grad_loss(w):
            return 2 * (w - 3.0)

        w_vals = np.linspace(-2, 9, 400)
        w = -1.5  # starting point

        history_w = [w]
        history_L = [loss(w)]
        for _ in range(n_steps):
            w = w - lr * grad_loss(w)
            history_w.append(w)
            history_L.append(loss(w))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Loss surface + GD path
        axes[0].plot(w_vals, loss(w_vals), "gray", linewidth=2, label="L(w) = (w-3)²")
        axes[0].plot(history_w, history_L, "o-", color="steelblue",
                     linewidth=2, markersize=6, label=f"GD path (LR={lr})")
        axes[0].scatter(history_w[0], history_L[0], c="green", s=150, zorder=5, label="Start")
        axes[0].scatter(history_w[-1], history_L[-1], c="red", s=150, marker="*", zorder=5, label="End")
        axes[0].axvline(3.0, color="black", linestyle="--", alpha=0.5, label="Optimum (w*=3)")
        axes[0].set_xlabel("w", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].set_title(f"Loss Landscape  (LR = {lr:.4f})", fontsize=12, fontweight="bold")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Loss vs step
        axes[1].semilogy(range(len(history_L)), history_L, "b-o", linewidth=2, markersize=4)
        axes[1].set_xlabel("Step", fontsize=12)
        axes[1].set_ylabel("Loss (log scale)", fontsize=12)
        axes[1].set_title("Convergence Curve", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        # Diagnose
        final_loss = history_L[-1]
        if final_loss > 100:
            status = "🚨 Diverging! LR too large."
        elif final_loss > 0.01:
            status = "🐢 Converging slowly. Try larger LR."
        else:
            status = "✅ Converged!"

        axes[1].set_title(f"Convergence  |  {status}", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.show()

    interact(
        update,
        lr=widgets.FloatLogSlider(min=-3, max=0.5, step=0.1, value=0.1,
                                   description="LR:", readout_format=".4f"),
        n_steps=widgets.IntSlider(min=5, max=50, value=20, description="Steps:")
    )


if __name__ == "__main__":
    print("widgets.py loaded OK ✓")
    print("Note: Widgets require ipywidgets and a Jupyter/Colab environment.")
    print("Available widgets:")
    print("  • activation_explorer_widget()")
    print("  • forward_pass_widget()")
    print("  • network_builder_widget()")
    print("  • training_playground_widget()")
    print("  • initialization_widget()")
    print("  • learning_rate_widget()")
