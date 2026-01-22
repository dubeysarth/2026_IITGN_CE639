"""
CE639 Lecture 05: Vector Calculus and Backpropagation - Helper Utilities
=========================================================================

This package contains helper functions for the Vector Calculus notebook,
organized into modular components for clean, maintainable code.

Modules:
    - calculus: Derivatives, gradients, Jacobians
    - computation_graph: DAG for autodiff
    - autodiff: Forward and backward-mode AD
    - neural_network: Simple NN with backprop
    - visualizations: Gradient fields, computation graphs
    - widgets: Interactive exploration tools
    - ce_examples: Civil Engineering applications
"""

from .calculus import (
    derivative,
    partial_derivative,
    gradient,
    jacobian,
    hessian,
    directional_derivative
)

from .computation_graph import (
    Node,
    ComputationGraph,
    forward_pass,
    backward_pass
)

from .autodiff import (
    Variable,
    forward_mode_ad,
    backward_mode_ad
)

from .neural_network import (
    Layer,
    NeuralNetwork,
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative
)

from .visualizations import (
    plot_gradient_field,
    plot_computation_graph,
    animate_backprop,
    plot_gradient_descent_3d,
    plot_loss_during_training,
    plot_jacobian_heatmap
)

from .widgets import (
    gradient_explorer_widget,
    chain_rule_widget,
    backprop_widget,
    neural_network_widget
)

from .ce_examples import (
    structural_stress_gradient,
    deflection_sensitivity,
    cost_function_gradient
)

__all__ = [
    # Calculus
    'derivative',
    'partial_derivative',
    'gradient',
    'jacobian',
    'hessian',
    'directional_derivative',
    # Computation Graph
    'Node',
    'ComputationGraph',
    'forward_pass',
    'backward_pass',
    # Autodiff
    'Variable',
    'forward_mode_ad',
    'backward_mode_ad',
    # Neural Network
    'Layer',
    'NeuralNetwork',
    'sigmoid',
    'sigmoid_derivative',
    'relu',
    'relu_derivative',
    # Visualizations
    'plot_gradient_field',
    'plot_computation_graph',
    'animate_backprop',
    'plot_gradient_descent_3d',
    'plot_loss_during_training',
    'plot_jacobian_heatmap',
    # Widgets
    'gradient_explorer_widget',
    'chain_rule_widget',
    'backprop_widget',
    'neural_network_widget',
    # CE Examples
    'structural_stress_gradient',
    'deflection_sensitivity',
    'cost_function_gradient'
]
