"""
Interactive Widgets for Dimensionality Reduction

This module provides ipywidgets-based interactive exploration
tools for PCA, autoencoders, and curse of dimensionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def pca_widget():
    """
    Interactive PCA exploration widget.
    
    Allows exploration of:
    - Number of components
    - Data characteristics
    - Reconstruction quality
    """
    from .pca import pca_svd, transform_pca, inverse_transform_pca
    from .visualizations import plot_pca_2d
    
    out = widgets.Output()
    
    n_components_slider = widgets.IntSlider(value=2, min=1, max=5, step=1,
                                            description='N Components:', continuous_update=False)
    n_features_slider = widgets.IntSlider(value=10, min=5, max=20, step=1,
                                          description='N Features:', continuous_update=False)
    n_samples_slider = widgets.IntSlider(value=200, min=50, max=500, step=50,
                                         description='N Samples:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(n_components, n_features, n_samples, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate correlated data
            mean = np.zeros(n_features)
            cov = np.random.rand(n_features, n_features)
            cov = cov @ cov.T  # Make positive definite
            X = np.random.multivariate_normal(mean, cov, n_samples)
            
            # PCA
            pca_result = pca_svd(X, n_components=n_components)
            
            # Transform and reconstruct
            Z_PC = transform_pca(X, pca_result)
            X_reconstructed = inverse_transform_pca(Z_PC, pca_result)
            
            # Reconstruction error
            mse = np.mean((X - X_reconstructed) ** 2)
            
            # Variance explained
            var_explained = np.sum(pca_result['explained_variance_ratio'])
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scree plot
            ax = axes[0]
            cumsum_var = np.cumsum(pca_result['explained_variance_ratio'])
            ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
            ax.axvline(n_components, color='red', linestyle='--', linewidth=2,
                      label=f'{n_components} components')
            ax.axhline(var_explained, color='green', linestyle='--', linewidth=2,
                      label=f'{var_explained*100:.1f}% variance')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Variance Explained')
            ax.set_title('Scree Plot', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Reconstruction quality
            ax = axes[1]
            sample_idx = 0
            ax.plot(X[sample_idx], 'b-', linewidth=2, label='Original')
            ax.plot(X_reconstructed[sample_idx], 'r--', linewidth=2, label='Reconstructed')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Value')
            ax.set_title(f'Reconstruction (MSE: {mse:.4f})', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"📊 Variance Explained: {var_explained*100:.2f}%")
            print(f"📉 Reconstruction MSE: {mse:.4f}")
    
    interactive = widgets.interactive(update, n_components=n_components_slider,
                                       n_features=n_features_slider, n_samples=n_samples_slider,
                                       seed=seed_slider)
    
    controls = widgets.VBox([n_components_slider, n_features_slider, n_samples_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(n_components_slider.value, n_features_slider.value, 
           n_samples_slider.value, seed_slider.value)
    
    return interactive


def autoencoder_widget():
    """
    Interactive autoencoder exploration widget.
    """
    from .autoencoders import Autoencoder, train_autoencoder
    
    out = widgets.Output()
    
    latent_dim_slider = widgets.IntSlider(value=2, min=1, max=5, step=1,
                                          description='Latent Dim:', continuous_update=False)
    hidden_dim_slider = widgets.IntSlider(value=8, min=4, max=16, step=2,
                                          description='Hidden Dim:', continuous_update=False)
    epochs_slider = widgets.IntSlider(value=50, min=10, max=200, step=10,
                                      description='Epochs:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(latent_dim, hidden_dim, epochs, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate data
            n_samples, input_dim = 200, 10
            mean = np.zeros(input_dim)
            cov = np.random.rand(input_dim, input_dim)
            cov = cov @ cov.T
            X = np.random.multivariate_normal(mean, cov, n_samples)
            
            # Standardize
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            # Train autoencoder
            autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim, random_state=seed)
            history = train_autoencoder(autoencoder, X, epochs=epochs, 
                                       learning_rate=0.01, verbose=False)
            
            # Reconstruct
            X_reconstructed, Z = autoencoder.forward(X)
            mse = np.mean((X - X_reconstructed) ** 2)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Training loss
            ax = axes[0]
            ax.plot(history['loss_history'], 'b-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.set_title('Training Loss', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Reconstruction
            ax = axes[1]
            sample_idx = 0
            ax.plot(X[sample_idx], 'b-', linewidth=2, label='Original')
            ax.plot(X_reconstructed[sample_idx], 'r--', linewidth=2, label='Reconstructed')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Value')
            ax.set_title(f'Reconstruction (MSE: {mse:.4f})', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print(f"📉 Final Loss: {history['loss_history'][-1]:.4f}")
            print(f"📊 Reconstruction MSE: {mse:.4f}")
    
    interactive = widgets.interactive(update, latent_dim=latent_dim_slider,
                                       hidden_dim=hidden_dim_slider, epochs=epochs_slider,
                                       seed=seed_slider)
    
    controls = widgets.VBox([latent_dim_slider, hidden_dim_slider, epochs_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(latent_dim_slider.value, hidden_dim_slider.value, 
           epochs_slider.value, seed_slider.value)
    
    return interactive


def curse_of_dim_widget():
    """
    Interactive curse of dimensionality demonstration.
    """
    from .curse_of_dimensionality import distance_concentration_demo
    
    out = widgets.Output()
    
    max_dim_slider = widgets.IntSlider(value=30, min=10, max=100, step=10,
                                       description='Max Dim:', continuous_update=False)
    n_points_slider = widgets.IntSlider(value=500, min=100, max=2000, step=100,
                                        description='N Points:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(max_dim, n_points, seed):
        with out:
            clear_output(wait=True)
            
            result = distance_concentration_demo(n_points=n_points, 
                                                 dimensions=range(1, max_dim + 1, 2),
                                                 random_state=seed)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Distances
            ax = axes[0]
            ax.plot(result['dimensions'], result['min_distances'], 'b-', 
                   linewidth=2, marker='o', label='Min')
            ax.plot(result['dimensions'], result['max_distances'], 'r-', 
                   linewidth=2, marker='s', label='Max')
            ax.plot(result['dimensions'], result['mean_distances'], 'g--', 
                   linewidth=2, marker='^', label='Mean')
            ax.set_xlabel('Dimensions')
            ax.set_ylabel('Distance')
            ax.set_title('Distance vs Dimensionality', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ratio
            ax = axes[1]
            ax.plot(result['dimensions'], result['ratio'], 'purple', 
                   linewidth=3, marker='o')
            ax.axhline(1.0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Dimensions')
            ax.set_ylabel('Min/Max Ratio')
            ax.set_title('Distance Concentration (→ 1)', fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            final_ratio = result['ratio'][-1]
            print(f"📊 Final Min/Max Ratio (D={max_dim}): {final_ratio:.3f}")
            print(f"⚠️ Distances become indistinguishable in high dimensions!")
    
    interactive = widgets.interactive(update, max_dim=max_dim_slider,
                                       n_points=n_points_slider, seed=seed_slider)
    
    controls = widgets.VBox([max_dim_slider, n_points_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(max_dim_slider.value, n_points_slider.value, seed_slider.value)
    
    return interactive


def interpretability_widget():
    """
    Interactive interpretability exploration.
    """
    from .pca import pca_svd
    from .interpretability import feature_importance_from_pca
    
    out = widgets.Output()
    
    n_components_slider = widgets.IntSlider(value=3, min=2, max=8, step=1,
                                            description='N Components:', continuous_update=False)
    n_features_slider = widgets.IntSlider(value=10, min=5, max=15, step=1,
                                          description='N Features:', continuous_update=False)
    seed_slider = widgets.IntSlider(value=42, min=0, max=100, step=1,
                                    description='Seed:', continuous_update=False)
    
    def update(n_components, n_features, seed):
        with out:
            clear_output(wait=True)
            
            np.random.seed(seed)
            
            # Generate data
            n_samples = 200
            mean = np.zeros(n_features)
            cov = np.random.rand(n_features, n_features)
            cov = cov @ cov.T
            X = np.random.multivariate_normal(mean, cov, n_samples)
            
            # PCA
            pca_result = pca_svd(X, n_components=n_components)
            
            # Feature importance
            importance_result = feature_importance_from_pca(pca_result)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loadings
            ax = axes[0]
            components = pca_result['components']
            x = np.arange(n_features)
            width = 0.8 / n_components
            colors = plt.cm.Set2(np.linspace(0, 1, n_components))
            
            for i in range(n_components):
                offset = (i - n_components/2) * width
                ax.bar(x + offset, components[i], width, label=f'PC{i+1}', 
                      color=colors[i], edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Loading')
            ax.set_title('PCA Loadings', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(0, color='black', linewidth=0.8)
            
            # Feature importance
            ax = axes[1]
            importance = importance_result['importance']
            ax.bar(range(n_features), importance, color='steelblue', edgecolor='black')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance Score')
            ax.set_title('Feature Importance from PCA', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
            
            print("📊 Top 3 Most Important Features:")
            for i, (feat, score) in enumerate(importance_result['ranking'][:3]):
                print(f"  {i+1}. {feat}: {score:.3f}")
    
    interactive = widgets.interactive(update, n_components=n_components_slider,
                                       n_features=n_features_slider, seed=seed_slider)
    
    controls = widgets.VBox([n_components_slider, n_features_slider, seed_slider])
    display(widgets.HBox([controls, out]))
    
    update(n_components_slider.value, n_features_slider.value, seed_slider.value)
    
    return interactive
