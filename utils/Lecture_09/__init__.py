"""
CE639 Lecture 09: Dimensionality Reduction - Helper Utilities
==============================================================

This package contains helper functions for the Dimensionality Reduction notebook,
organized into modular components for clean, maintainable code.

Modules:
    - pca: Principal Component Analysis (covariance + SVD methods)
    - autoencoders: Simple autoencoder implementations
    - curse_of_dimensionality: Demonstrations of high-dim problems
    - interpretability: PC loadings, feature importance
    - visualizations: Scree plots, latent space, loadings
    - widgets: Interactive exploration tools
    - ce_examples: Civil Engineering dimensionality reduction datasets
"""

# PCA
from .pca import (
    standardize_data,
    pca_covariance,
    pca_svd,
    explained_variance_ratio,
    transform_pca,
    inverse_transform_pca,
    pca_step_by_step
)

# Autoencoders
from .autoencoders import (
    LinearAutoencoder,
    Autoencoder,
    train_autoencoder,
    encode,
    decode,
    reconstruction_error
)

# Curse of Dimensionality
from .curse_of_dimensionality import (
    distance_concentration_demo,
    volume_growth_demo,
    sampling_requirement,
    nearest_neighbor_degradation
)

# Interpretability
from .interpretability import (
    pca_loadings_plot,
    feature_importance_from_pca,
    compare_original_vs_compressed
)

# Visualizations
from .visualizations import (
    plot_scree,
    plot_pca_2d,
    plot_pca_3d,
    plot_autoencoder_reconstruction,
    plot_latent_space,
    plot_distance_concentration,
    plot_loadings_heatmap
)

# Widgets
from .widgets import (
    pca_widget,
    autoencoder_widget,
    curse_of_dim_widget,
    interpretability_widget
)

# CE Examples
from .ce_examples import (
    bridge_sensor_data,
    material_spectral_data,
    traffic_multivariate_data,
    structural_mode_shapes,
    high_dimensional_ce_data
)

__all__ = [
    # PCA
    'standardize_data',
    'pca_covariance',
    'pca_svd',
    'explained_variance_ratio',
    'transform_pca',
    'inverse_transform_pca',
    'pca_step_by_step',
    # Autoencoders
    'LinearAutoencoder',
    'Autoencoder',
    'train_autoencoder',
    'encode',
    'decode',
    'reconstruction_error',
    # Curse of Dimensionality
    'distance_concentration_demo',
    'volume_growth_demo',
    'sampling_requirement',
    'nearest_neighbor_degradation',
    # Interpretability
    'pca_loadings_plot',
    'feature_importance_from_pca',
    'compare_original_vs_compressed',
    # Visualizations
    'plot_scree',
    'plot_pca_2d',
    'plot_pca_3d',
    'plot_autoencoder_reconstruction',
    'plot_latent_space',
    'plot_distance_concentration',
    'plot_loadings_heatmap',
    # Widgets
    'pca_widget',
    'autoencoder_widget',
    'curse_of_dim_widget',
    'interpretability_widget',
    # CE Examples
    'bridge_sensor_data',
    'material_spectral_data',
    'traffic_multivariate_data',
    'structural_mode_shapes',
    'high_dimensional_ce_data',
]
