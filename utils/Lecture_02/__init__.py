"""
CE639 Lecture 02: Linear Algebra for ML - Helper Utilities
===========================================================

This package contains helper functions for the Linear Algebra notebook,
organized into modular components for clean, maintainable code.

Modules:
    - visualizations: Plotting and animation functions
    - widgets: Interactive ipywidgets for exploration
    - ce_examples: Civil Engineering application examples
    - numerical: Numerical computing demonstrations
"""

from .visualizations import (
    plot_vector_2d,
    plot_vector_3d,
    plot_matrix_heatmap,
    plot_norm_unit_balls,
    animate_projection,
    animate_eigen_transform,
    plot_svd_compression
)

from .widgets import (
    vector_scaling_widget,
    norm_explorer_widget,
    projection_widget,
    svd_rank_widget,
    condition_number_widget
)

from .ce_examples import (
    create_stiffness_matrix,
    create_vibration_system,
    simulate_structural_modes,
    generate_sensor_data,
    create_force_decomposition
)

from .numerical import (
    demonstrate_floating_point,
    memory_usage_table,
    condition_number_demo,
    near_singular_demo
)

__all__ = [
    # Visualizations
    'plot_vector_2d',
    'plot_vector_3d',
    'plot_matrix_heatmap',
    'plot_norm_unit_balls',
    'animate_projection',
    'animate_eigen_transform',
    'plot_svd_compression',
    # Widgets
    'vector_scaling_widget',
    'norm_explorer_widget',
    'projection_widget',
    'svd_rank_widget',
    'condition_number_widget',
    # CE Examples
    'create_stiffness_matrix',
    'create_vibration_system',
    'simulate_structural_modes',
    'generate_sensor_data',
    'create_force_decomposition',
    # Numerical
    'demonstrate_floating_point',
    'memory_usage_table',
    'condition_number_demo',
    'near_singular_demo'
]
