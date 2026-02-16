"""
CE639 Lecture 07: Classification - Helper Utilities
====================================================

This package contains helper functions for the Classification notebook,
organized into modular components for clean, maintainable code.

Modules:
    - logistic_regression: Sigmoid, softmax, cross-entropy, gradient descent
    - metrics: Confusion matrix, precision/recall/F1, ROC/PR curves, AUC
    - class_imbalance: SMOTE, over/undersampling, weighted loss
    - decision_boundaries: 2D boundary visualization and animation
    - visualizations: Loss landscapes, training history plots
    - widgets: Interactive exploration tools
    - ce_examples: Civil Engineering classification datasets
"""

# Logistic Regression
from .logistic_regression import (
    sigmoid,
    softmax,
    cross_entropy_loss,
    categorical_cross_entropy_loss,
    logistic_regression_gd,
    logistic_predict,
    logistic_predict_class,
    softmax_regression_gd,
    softmax_predict,
    softmax_predict_class
)

# Metrics
from .metrics import (
    confusion_matrix,
    accuracy,
    precision,
    recall,
    specificity,
    f1_score,
    compute_all_classification_metrics,
    roc_curve,
    pr_curve,
    auc,
    roc_auc_score,
    pr_auc_score,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)

# Class Imbalance
from .class_imbalance import (
    random_oversample,
    random_undersample,
    smote,
    weighted_cross_entropy,
    compute_class_weights,
    generate_imbalanced_data,
    plot_class_distribution
)

# Decision Boundaries
from .decision_boundaries import (
    create_mesh_grid,
    plot_decision_boundary_2d,
    plot_sigmoid_curve,
    animate_decision_boundary,
    plot_multiclass_boundaries,
    plot_probability_surface_3d
)

# Visualizations
from .visualizations import (
    plot_sigmoid_vs_linear,
    plot_softmax_demo,
    plot_loss_landscape_classification,
    plot_training_history,
    plot_threshold_impact,
    plot_class_separation
)

# CE Examples
from .ce_examples import (
    crack_detection_data,
    structural_failure_data,
    soil_classification_data,
    flood_prediction_data,
    pavement_condition_data,
    generate_linearly_separable_data
)

# Widgets
from .widgets import (
    logistic_regression_widget,
    decision_boundary_widget,
    class_imbalance_widget,
    confusion_matrix_widget,
    roc_curve_widget,
    multiclass_widget
)

__all__ = [
    # Logistic Regression
    'sigmoid',
    'softmax',
    'cross_entropy_loss',
    'categorical_cross_entropy_loss',
    'logistic_regression_gd',
    'logistic_predict',
    'logistic_predict_class',
    'softmax_regression_gd',
    'softmax_predict',
    'softmax_predict_class',
    # Metrics
    'confusion_matrix',
    'accuracy',
    'precision',
    'recall',
    'specificity',
    'f1_score',
    'compute_all_classification_metrics',
    'roc_curve',
    'pr_curve',
    'auc',
    'roc_auc_score',
    'pr_auc_score',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    # Class Imbalance
    'random_oversample',
    'random_undersample',
    'smote',
    'weighted_cross_entropy',
    'compute_class_weights',
    'generate_imbalanced_data',
    'plot_class_distribution',
    # Decision Boundaries
    'create_mesh_grid',
    'plot_decision_boundary_2d',
    'plot_sigmoid_curve',
    'animate_decision_boundary',
    'plot_multiclass_boundaries',
    'plot_probability_surface_3d',
    # Visualizations
    'plot_sigmoid_vs_linear',
    'plot_softmax_demo',
    'plot_loss_landscape_classification',
    'plot_training_history',
    'plot_threshold_impact',
    'plot_class_separation',
    # CE Examples
    'crack_detection_data',
    'structural_failure_data',
    'soil_classification_data',
    'flood_prediction_data',
    'pavement_condition_data',
    'generate_linearly_separable_data',
    # Widgets
    'logistic_regression_widget',
    'decision_boundary_widget',
    'class_imbalance_widget',
    'confusion_matrix_widget',
    'roc_curve_widget',
    'multiclass_widget',
]
