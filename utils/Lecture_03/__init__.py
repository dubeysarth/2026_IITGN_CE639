"""
CE639 Lecture 03: Probability & Statistics - Helper Utilities
==============================================================

This package contains helper functions for the Probability & Statistics notebook,
organized into modular components for clean, maintainable code.

Modules:
    - distributions: PDF/CDF functions for various distributions
    - visualizations: Plotting and animation functions
    - widgets: Interactive ipywidgets for exploration
    - ce_examples: Civil Engineering application examples
"""

from .distributions import (
    uniform_pdf, uniform_cdf,
    normal_pdf, normal_cdf,
    gumbel_pdf, gumbel_cdf,
    gev_pdf, gev_cdf,
    compute_moments
)

from .visualizations import (
    plot_pmf, plot_pdf, plot_cdf,
    plot_distribution_comparison,
    animate_sampling,
    plot_covariance_matrix,
    plot_correlation_scatter,
    animate_clt
)

from .widgets import (
    distribution_explorer_widget,
    sampling_widget,
    covariance_widget,
    extreme_value_widget,
    moments_explorer_widget
)

from .ce_examples import (
    generate_concrete_strength_data,
    generate_flood_data,
    generate_sensor_noise,
    generate_load_data,
    generate_wind_speed_data,
    generate_correlated_data
)

__all__ = [
    # Distributions
    'uniform_pdf', 'uniform_cdf',
    'normal_pdf', 'normal_cdf',
    'gumbel_pdf', 'gumbel_cdf',
    'gev_pdf', 'gev_cdf',
    'compute_moments',
    # Visualizations
    'plot_pmf', 'plot_pdf', 'plot_cdf',
    'plot_distribution_comparison',
    'animate_sampling',
    'plot_covariance_matrix',
    'plot_correlation_scatter',
    'animate_clt',
    # Widgets
    'distribution_explorer_widget',
    'sampling_widget',
    'covariance_widget',
    'extreme_value_widget',
    'moments_explorer_widget',
    # CE Examples
    'generate_concrete_strength_data',
    'generate_flood_data',
    'generate_sensor_noise',
    'generate_load_data',
    'generate_wind_speed_data',
    'generate_correlated_data'
]
