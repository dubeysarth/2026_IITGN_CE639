"""
CE639 Lecture 04: Optimization and Loss Landscapes - Helper Utilities
======================================================================

This package contains helper functions for the Optimization notebook,
organized into modular components for clean, maintainable code.

Modules:
    - optimizers: Gradient descent variants and learning rate schedules
    - loss_functions: Loss functions with regularization
    - visualizations: Loss landscape plots and animations
    - widgets: Interactive ipywidgets for exploration
    - ce_examples: Civil Engineering optimization examples
"""

from .optimizers import (
    gradient_descent,
    stochastic_gradient_descent,
    adam_optimizer,
    compute_gradient,
    learning_rate_schedule
)

from .loss_functions import (
    mse_loss,
    mse_gradient,
    cross_entropy_loss,
    mse_with_l1_reg,
    mse_with_l1_gradient,
    mse_with_l2_reg,
    mse_with_l2_gradient,
    mse_with_elastic_net,
    quadratic_bowl,
    rosenbrock,
    himmelblau
)

from .visualizations import (
    plot_loss_landscape_2d,
    plot_loss_landscape_3d,
    animate_gradient_descent,
    plot_learning_rate_comparison,
    plot_convex_vs_nonconvex,
    plot_bias_variance_tradeoff,
    plot_regularization_paths,
    plot_gradient_field
)

from .widgets import (
    learning_rate_widget,
    gd_comparison_widget,
    convexity_widget,
    bias_variance_widget,
    regularization_widget
)

from .ce_examples import (
    create_structural_optimization,
    create_cost_function,
    create_regression_data,
    simulate_overfitting,
    demonstrate_dropout
)

__all__ = [
    # Optimizers
    'gradient_descent',
    'stochastic_gradient_descent',
    'adam_optimizer',
    'compute_gradient',
    'learning_rate_schedule',
    # Loss Functions
    'mse_loss',
    'mse_gradient',
    'cross_entropy_loss',
    'mse_with_l1_reg',
    'mse_with_l1_gradient',
    'mse_with_l2_reg',
    'mse_with_l2_gradient',
    'mse_with_elastic_net',
    'quadratic_bowl',
    'rosenbrock',
    'himmelblau',
    # Visualizations
    'plot_loss_landscape_2d',
    'plot_loss_landscape_3d',
    'animate_gradient_descent',
    'plot_learning_rate_comparison',
    'plot_convex_vs_nonconvex',
    'plot_bias_variance_tradeoff',
    'plot_regularization_paths',
    'plot_gradient_field',
    # Widgets
    'learning_rate_widget',
    'gd_comparison_widget',
    'convexity_widget',
    'bias_variance_widget',
    'regularization_widget',
    # CE Examples
    'create_structural_optimization',
    'create_cost_function',
    'create_regression_data',
    'simulate_overfitting',
    'demonstrate_dropout'
]
