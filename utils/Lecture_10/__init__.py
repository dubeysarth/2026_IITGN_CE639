"""
Lecture 10: Feedforward Neural Networks
Helper utilities for CE 639: AI for Civil Engineering

This package provides from-scratch (NumPy) and PyTorch implementations of
feedforward neural network components, together with CE-specific datasets,
rich visualisations, and interactive widgets.

Usage
-----
    import sys; sys.path.append('..')
    from utils.Lecture_10 import (
        sigmoid, relu, tanh_act, softmax,
        NumpyFNN, forward_pass_step_by_step,
        SimpleFNN, BeamDeflectionNet,
        train_numpy_fnn, train_pytorch_fnn,
        generate_beam_deflection_dataset,
        plot_activation_gallery, plot_network_diagram,
        activation_explorer_widget, training_playground_widget,
        ...
    )
"""

# ── Perceptron / single neuron ────────────────────────────────────────────────
from .perceptron import (
    perceptron_forward,
    perceptron_decision_boundary,
    neuron_activation_region,
    single_neuron_gradient_step,
    single_neuron_gradient_descent,
    perceptron_train,
    relu_piecewise_boundary,
)

# ── Activation functions ──────────────────────────────────────────────────────
from .activations import (
    sigmoid,
    sigmoid_derivative,
    tanh_act,
    tanh_derivative,
    relu,
    relu_derivative,
    leaky_relu,
    leaky_relu_derivative,
    elu,
    elu_derivative,
    gelu,
    gelu_derivative,
    softmax,
    get_activation,
    get_derivative,
    activation_summary_table,
    compute_saturation_fraction,
    compute_dead_relu_fraction,
    safe_log,
)

# ── NumPy FNN ─────────────────────────────────────────────────────────────────
from .network import (
    NumpyFNN,
    init_weights,
    forward_pass_step_by_step,
    count_parameters,
    demonstrate_linear_collapse,
)

# ── Training utilities ────────────────────────────────────────────────────────
from .training import (
    mse_loss,
    mse_grad,
    binary_cross_entropy_loss,
    cross_entropy_loss,
    train_numpy_fnn,
    train_pytorch_fnn,
    evaluate_pytorch,
    plot_training_history,
    plot_decision_regions,
)

# ── PyTorch architectures ─────────────────────────────────────────────────────
from .architectures import (
    SimpleFNN,
    BeamDeflectionNet,
    ConcreteStrengthNet,
    TrafficFlowNet,
    build_fnn,
    count_parameters as count_torch_parameters,
    model_summary,
)

# ── CE datasets ───────────────────────────────────────────────────────────────
from .ce_datasets import (
    generate_beam_deflection_dataset,
    generate_concrete_strength_dataset,
    generate_traffic_flow_dataset,
    generate_xor_dataset,
    generate_spiral_dataset,
    generate_regression_1d,
)

# ── Visualisations ────────────────────────────────────────────────────────────
from .visualizations import (
    plot_activation_gallery,
    plot_network_diagram,
    plot_gradient_flow,
    plot_init_comparison,
    plot_regularization_comparison,
    plot_loss_landscape_2d,
    plot_depth_vs_width,
    plot_batch_norm_effect,
)

# ── Interactive widgets ───────────────────────────────────────────────────────
from .widgets import (
    activation_explorer_widget,
    forward_pass_widget,
    network_builder_widget,
    training_playground_widget,
    initialization_widget,
    learning_rate_widget,
)

__all__ = [
    # Perceptron
    "perceptron_forward", "perceptron_decision_boundary", "neuron_activation_region",
    "single_neuron_gradient_step", "single_neuron_gradient_descent",
    "perceptron_train", "relu_piecewise_boundary",
    # Activations
    "sigmoid", "sigmoid_derivative", "tanh_act", "tanh_derivative",
    "relu", "relu_derivative", "leaky_relu", "leaky_relu_derivative",
    "elu", "elu_derivative", "gelu", "gelu_derivative", "softmax",
    "get_activation", "get_derivative", "activation_summary_table",
    "compute_saturation_fraction", "compute_dead_relu_fraction", "safe_log",
    # Network
    "NumpyFNN", "init_weights", "forward_pass_step_by_step",
    "count_parameters", "demonstrate_linear_collapse",
    # Training
    "mse_loss", "mse_grad", "binary_cross_entropy_loss", "cross_entropy_loss",
    "train_numpy_fnn", "train_pytorch_fnn", "evaluate_pytorch",
    "plot_training_history", "plot_decision_regions",
    # Architectures
    "SimpleFNN", "BeamDeflectionNet", "ConcreteStrengthNet", "TrafficFlowNet",
    "build_fnn", "count_torch_parameters", "model_summary",
    # Datasets
    "generate_beam_deflection_dataset", "generate_concrete_strength_dataset",
    "generate_traffic_flow_dataset", "generate_xor_dataset",
    "generate_spiral_dataset", "generate_regression_1d",
    # Visualisations
    "plot_activation_gallery", "plot_network_diagram", "plot_gradient_flow",
    "plot_init_comparison", "plot_regularization_comparison",
    "plot_loss_landscape_2d", "plot_depth_vs_width", "plot_batch_norm_effect",
    # Widgets
    "activation_explorer_widget", "forward_pass_widget", "network_builder_widget",
    "training_playground_widget", "initialization_widget", "learning_rate_widget",
]
