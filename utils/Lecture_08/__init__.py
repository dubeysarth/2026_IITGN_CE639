"""
CE639 Lecture 08: Clustering & Unsupervised Learning - Helper Utilities
========================================================================

This package contains helper functions for the Clustering notebook,
organized into modular components for clean, maintainable code.

Modules:
    - kmeans: K-Means clustering algorithm
    - hierarchical: Agglomerative hierarchical clustering
    - similarity_metrics: Distance and similarity functions
    - cluster_validation: Elbow method, silhouette score
    - regime_identification: Time-series regime detection
    - visualizations: Cluster plots, dendrograms
    - widgets: Interactive exploration tools
    - ce_examples: Civil Engineering clustering datasets
"""

# K-Means
from .kmeans import (
    kmeans,
    kmeans_plusplus_init,
    compute_wcss,
    assign_clusters,
    update_centroids,
    kmeans_step_by_step
)

# Hierarchical Clustering
from .hierarchical import (
    agglomerative_clustering,
    compute_linkage_matrix,
    single_linkage,
    complete_linkage,
    average_linkage,
    ward_linkage,
    cut_dendrogram
)

# Similarity Metrics
from .similarity_metrics import (
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
    cosine_similarity,
    cosine_distance,
    hamming_distance,
    jaccard_similarity,
    pairwise_distances
)

# Cluster Validation
from .cluster_validation import (
    silhouette_score,
    silhouette_samples,
    elbow_method,
    find_optimal_k
)

# Regime Identification
from .regime_identification import (
    extract_window_features,
    detect_change_points,
    cluster_based_regime_detection,
    plot_regime_timeline
)

# Visualizations
from .visualizations import (
    plot_clusters_2d,
    plot_clusters_3d,
    plot_dendrogram,
    plot_elbow,
    plot_silhouette,
    plot_kmeans_animation,
    plot_linkage_comparison
)

# Widgets
from .widgets import (
    kmeans_widget,
    hierarchical_widget,
    similarity_widget,
    elbow_silhouette_widget
)

# CE Examples
from .ce_examples import (
    sensor_clustering_data,
    traffic_flow_data,
    material_property_data,
    shm_vibration_data,
    generate_blob_data
)

__all__ = [
    # K-Means
    'kmeans',
    'kmeans_plusplus_init',
    'compute_wcss',
    'assign_clusters',
    'update_centroids',
    'kmeans_step_by_step',
    # Hierarchical
    'agglomerative_clustering',
    'compute_linkage_matrix',
    'single_linkage',
    'complete_linkage',
    'average_linkage',
    'ward_linkage',
    'cut_dendrogram',
    # Similarity Metrics
    'euclidean_distance',
    'manhattan_distance',
    'minkowski_distance',
    'cosine_similarity',
    'cosine_distance',
    'hamming_distance',
    'jaccard_similarity',
    'pairwise_distances',
    # Cluster Validation
    'silhouette_score',
    'silhouette_samples',
    'elbow_method',
    'find_optimal_k',
    # Regime Identification
    'extract_window_features',
    'detect_change_points',
    'cluster_based_regime_detection',
    'plot_regime_timeline',
    # Visualizations
    'plot_clusters_2d',
    'plot_clusters_3d',
    'plot_dendrogram',
    'plot_elbow',
    'plot_silhouette',
    'plot_kmeans_animation',
    'plot_linkage_comparison',
    # Widgets
    'kmeans_widget',
    'hierarchical_widget',
    'similarity_widget',
    'elbow_silhouette_widget',
    # CE Examples
    'sensor_clustering_data',
    'traffic_flow_data',
    'material_property_data',
    'shm_vibration_data',
    'generate_blob_data',
]
