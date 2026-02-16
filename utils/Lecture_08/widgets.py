"""
Interactive Widgets for Clustering

This module provides ipywidgets-based interactive exploration
tools for clustering concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def kmeans_widget():
    """
    Interactive K-Means clustering widget.
    
    Allows exploration of:
    - Number of clusters (K)
    - Initialization method
    - Number of data points
    - Cluster spread
    """
    from .kmeans import kmeans, kmeans_plusplus_init
    from .visualizations import plot_clusters_2d
    
    # Output area
    out = widgets.Output()
    
    # Sliders
    k_slider = widgets.IntSlider(value=3, min=2, max=8, step=1,
                                 description='K:', continuous_update=False)
    n_samples_slider = widgets.IntSlider(value=150, min=50, max=500, step=50,
                                         description='N Samples:', continuous_update=False)
    spread_slider = widgets.FloatSlider(value=1.0, min=0.3, max=3.0, step=0.1,
                                        description='Spread:', continuous_update=False)
    init_dropdown = widgets.Dropdown(options=['kmeans++', 'random'],
                                     value='kmeans++', description='Init:')
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(k, n_samples, spread, init, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate blob data
            centers = np.random.uniform(-5, 5, (k, 2))
            X = []
            for center in centers:
                cluster_points = np.random.normal(loc=center, scale=spread, size=(n_samples // k, 2))
                X.append(cluster_points)
            X = np.vstack(X)
            
            # Run K-Means
            result = kmeans(X, k, init=init, random_state=seed)
            
            fig, ax = plot_clusters_2d(X, result['labels'], result['centroids'],
                                       title=f'K-Means Clustering (K={k})')
            
            ax.text(0.02, 0.98, f"WCSS: {result['wcss']:.2f}\nIterations: {result['n_iterations']}",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
    
    # Link widgets
    interactive = widgets.interactive(update, k=k_slider, n_samples=n_samples_slider,
                                       spread=spread_slider, init=init_dropdown, seed=seed_slider)
    
    # Layout
    controls = widgets.VBox([k_slider, n_samples_slider, spread_slider, init_dropdown, seed_slider])
    display(widgets.HBox([controls, out]))
    
    # Initial update
    update(k_slider.value, n_samples_slider.value, spread_slider.value, 
           init_dropdown.value, seed_slider.value)
    
    return interactive


def hierarchical_widget():
    """
    Interactive hierarchical clustering widget.
    
    Allows exploration of:
    - Linkage method
    - Number of clusters
    - Data characteristics
    """
    from .hierarchical import agglomerative_clustering
    from .visualizations import plot_clusters_2d
    
    out = widgets.Output()
    
    n_clusters_slider = widgets.IntSlider(value=3, min=2, max=6, step=1,
                                          description='N Clusters:', continuous_update=False)
    method_dropdown = widgets.Dropdown(options=['single', 'complete', 'average', 'ward'],
                                       value='ward', description='Linkage:')
    n_samples_slider = widgets.IntSlider(value=80, min=30, max=150, step=10,
                                         description='N Samples:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(n_clusters, method, n_samples, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate data with different shapes
            n_per = n_samples // 3
            X1 = np.random.normal(loc=[0, 0], scale=0.8, size=(n_per, 2))
            X2 = np.random.normal(loc=[4, 4], scale=0.8, size=(n_per, 2))
            X3 = np.random.normal(loc=[0, 5], scale=1.2, size=(n_samples - 2*n_per, 2))
            X = np.vstack([X1, X2, X3])
            
            # Cluster
            result = agglomerative_clustering(X, n_clusters=n_clusters, method=method)
            
            fig, ax = plot_clusters_2d(X, result['labels'], centroids=None,
                                       title=f'{method.capitalize()} Linkage (K={n_clusters})')
            
            plt.tight_layout()
            plt.show()
    
    interactive = widgets.interactive(update, n_clusters=n_clusters_slider,
                                       method=method_dropdown, n_samples=n_samples_slider,
                                       seed=seed_slider)
    
    controls = widgets.VBox([n_clusters_slider, method_dropdown, n_samples_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(n_clusters_slider.value, method_dropdown.value, 
           n_samples_slider.value, seed_slider.value)
    
    return interactive


def similarity_widget():
    """
    Interactive widget to compare similarity metrics.
    
    Visualizes how different metrics measure distance
    between two points.
    """
    from .similarity_metrics import (euclidean_distance, manhattan_distance,
                                     minkowski_distance, cosine_similarity)
    
    out = widgets.Output()
    
    x1_slider = widgets.FloatSlider(value=1.0, min=-5, max=5, step=0.5,
                                    description='x₁:', continuous_update=False)
    y1_slider = widgets.FloatSlider(value=2.0, min=-5, max=5, step=0.5,
                                    description='y₁:', continuous_update=False)
    x2_slider = widgets.FloatSlider(value=4.0, min=-5, max=5, step=0.5,
                                    description='x₂:', continuous_update=False)
    y2_slider = widgets.FloatSlider(value=5.0, min=-5, max=5, step=0.5,
                                    description='y₂:', continuous_update=False)
    
    def update(x1, y1, x2, y2):
        with out:
            clear_output(wait=True)
            
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            
            # Compute metrics
            euclidean = euclidean_distance(p1, p2)
            manhattan = manhattan_distance(p1, p2)
            cosine = cosine_similarity(p1, p2)
            minkowski_3 = minkowski_distance(p1, p2, p=3)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: Visualization
            ax = axes[0]
            ax.scatter(*p1, s=200, c='blue', marker='o', edgecolors='black', 
                      linewidths=2, label=f'Point 1 ({x1:.1f}, {y1:.1f})')
            ax.scatter(*p2, s=200, c='red', marker='s', edgecolors='black', 
                      linewidths=2, label=f'Point 2 ({x2:.1f}, {y2:.1f})')
            
            # Draw Euclidean line
            ax.plot([x1, x2], [y1, y2], 'g--', linewidth=2, label=f'Euclidean: {euclidean:.2f}')
            
            # Draw Manhattan path
            ax.plot([x1, x2, x2], [y1, y1, y2], 'orange', linewidth=2, 
                   linestyle=':', label=f'Manhattan: {manhattan:.2f}')
            
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.axvline(0, color='gray', linewidth=0.5)
            ax.set_xlabel('X', fontsize=11)
            ax.set_ylabel('Y', fontsize=11)
            ax.set_title('Distance Visualization', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Right: Metrics comparison
            ax2 = axes[1]
            metrics = ['Euclidean', 'Manhattan', 'Minkowski\n(p=3)', 'Cosine\nSimilarity']
            values = [euclidean, manhattan, minkowski_3, cosine]
            colors = ['#2ecc71', '#e67e22', '#3498db', '#9b59b6']
            
            bars = ax2.barh(metrics, values, color=colors, edgecolor='black')
            ax2.set_xlabel('Value', fontsize=11)
            ax2.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
            
            for bar, val in zip(bars, values):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=10)
            
            ax2.set_xlim(0, max(values) * 1.3 + 0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.show()
    
    interactive = widgets.interactive(update, x1=x1_slider, y1=y1_slider,
                                       x2=x2_slider, y2=y2_slider)
    
    controls = widgets.VBox([x1_slider, y1_slider, x2_slider, y2_slider])
    display(widgets.HBox([controls, out]))
    
    update(x1_slider.value, y1_slider.value, x2_slider.value, y2_slider.value)
    
    return interactive


def elbow_silhouette_widget():
    """
    Interactive widget for elbow method and silhouette analysis.
    
    Helps students understand how to choose optimal K.
    """
    from .kmeans import kmeans
    from .cluster_validation import silhouette_score
    
    out = widgets.Output()
    
    n_samples_slider = widgets.IntSlider(value=150, min=50, max=300, step=50,
                                         description='N Samples:', continuous_update=False)
    true_k_slider = widgets.IntSlider(value=3, min=2, max=6, step=1,
                                      description='True K:', continuous_update=False)
    spread_slider = widgets.FloatSlider(value=1.0, min=0.3, max=2.0, step=0.1,
                                        description='Spread:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(n_samples, true_k, spread, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate blob data with true_k clusters
            centers = np.random.uniform(-5, 5, (true_k, 2))
            X = []
            for center in centers:
                cluster_points = np.random.normal(loc=center, scale=spread, 
                                                  size=(n_samples // true_k, 2))
                X.append(cluster_points)
            X = np.vstack(X)
            
            # Compute metrics for K = 1 to 8
            k_range = range(1, 9)
            wcss_values = []
            silhouette_values = []
            
            for k in k_range:
                result = kmeans(X, k, random_state=seed)
                wcss_values.append(result['wcss'])
                
                if k > 1:
                    sil = silhouette_score(X, result['labels'])
                else:
                    sil = -1  # Undefined for k=1
                silhouette_values.append(sil)
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Elbow plot
            ax1 = axes[0]
            ax1.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=10)
            ax1.axvline(true_k, color='red', linestyle='--', linewidth=2,
                       label=f'True K = {true_k}')
            ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax1.set_ylabel('WCSS', fontsize=11)
            ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
            ax1.set_xticks(list(k_range))
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Silhouette plot
            ax2 = axes[1]
            ax2.plot(k_range, silhouette_values, 'go-', linewidth=2, markersize=10)
            ax2.axvline(true_k, color='red', linestyle='--', linewidth=2,
                       label=f'True K = {true_k}')
            ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
            ax2.set_ylabel('Silhouette Score', fontsize=11)
            ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
            ax2.set_xticks(list(k_range))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Highlight optimal K
            valid_k = [k for k in k_range if k > 1]
            valid_sil = [silhouette_values[i] for i, k in enumerate(k_range) if k > 1]
            best_k = valid_k[np.argmax(valid_sil)]
            ax2.scatter([best_k], [max(valid_sil)], c='green', s=200, marker='*', 
                       zorder=10, label=f'Best K = {best_k}')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
    
    interactive = widgets.interactive(update, n_samples=n_samples_slider,
                                       true_k=true_k_slider, spread=spread_slider,
                                       seed=seed_slider)
    
    controls = widgets.VBox([n_samples_slider, true_k_slider, spread_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(n_samples_slider.value, true_k_slider.value, 
           spread_slider.value, seed_slider.value)
    
    return interactive
