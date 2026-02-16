"""
CE639 Lecture 06: Regression Models - Helper Utilities
=======================================================

This package contains helper functions for the Regression notebook,
organized into modular components for clean, maintainable code.

Modules:
    - linear_regression: OLS, Ridge, LASSO regression
    - polynomial_regression: Polynomial feature transformation
    - kernel_regression: Kernel methods (linear, polynomial, RBF)
    - error_metrics: MAE, MAPE, RMSE, NSE, R²
    - visualizations: Regression plots, residuals, comparisons
    - widgets: Interactive exploration tools
    - ce_examples: Civil Engineering datasets
"""

from .linear_regression import (
    ols_fit,
    ridge_fit,
    lasso_fit,
    linear_predict,
    linear_regression_gd,
    standardize_features,
    normalize_features
)

from .polynomial_regression import (
    polynomial_features,
    polynomial_fit,
    polynomial_predict
)

from .kernel_regression import (
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    kernel_matrix,
    kernel_regression_fit,
    kernel_regression_predict
)

from .error_metrics import (
    mae,
    mape,
    mse,
    rmse,
    nse,
    r_squared,
    compute_all_metrics
)

from .visualizations import (
    plot_regression_fit,
    plot_residuals,
    plot_kernel_comparison,
    plot_polynomial_degrees,
    plot_regularization_path,
    plot_error_metrics
)

from .widgets import (
    linear_regression_widget,
    polynomial_degree_widget,
    kernel_explorer_widget,
    regularization_widget,
    error_metric_widget
)

from .ce_examples import (
    concrete_strength_data,
    traffic_flow_data,
    pavement_deflection_data
)

__all__ = [
    # Linear Regression
    'ols_fit',
    'ridge_fit',
    'lasso_fit',
    'linear_predict',
    'linear_regression_gd',
    'standardize_features',
    'normalize_features',
    # Polynomial Regression
    'polynomial_features',
    'polynomial_fit',
    'polynomial_predict',
    # Kernel Regression
    'linear_kernel',
    'polynomial_kernel',
    'rbf_kernel',
    'kernel_matrix',
    'kernel_regression_fit',
    'kernel_regression_predict',
    # Error Metrics
    'mae',
    'mape',
    'mse',
    'rmse',
    'nse',
    'r_squared',
    'compute_all_metrics',
    # Visualizations
    'plot_regression_fit',
    'plot_residuals',
    'plot_kernel_comparison',
    'plot_polynomial_degrees',
    'plot_regularization_path',
    'plot_error_metrics',
    # Widgets
    'linear_regression_widget',
    'polynomial_degree_widget',
    'kernel_explorer_widget',
    'regularization_widget',
    'error_metric_widget',
    # CE Examples
    'concrete_strength_data',
    'traffic_flow_data',
    'pavement_deflection_data'
]
