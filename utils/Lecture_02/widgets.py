"""
Interactive widgets for exploring Linear Algebra concepts.

This module provides ipywidgets-based interactive controls for
vector scaling, norm exploration, projections, SVD, and conditioning.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, FloatSlider, IntSlider, fixed
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def vector_scaling_widget(initial_vector=[1, 2]):
    """
    Interactive widget to explore vector scaling.
    
    Parameters:
    -----------
    initial_vector : list
        Initial 2D vector to scale
    """
    def plot_scaled_vector(scale):
        vec = np.array(initial_vector)
        scaled_vec = scale * vec
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot original vector
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3,
                fc='blue', ec='blue', linewidth=2, label='Original', alpha=0.5)
        
        # Plot scaled vector
        ax.arrow(0, 0, scaled_vec[0], scaled_vec[1], head_width=0.2, head_length=0.3,
                fc='red', ec='red', linewidth=2, label=f'Scaled (×{scale:.1f})')
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        max_val = max(abs(scaled_vec[0]), abs(scaled_vec[1]), abs(vec[0]), abs(vec[1])) + 1
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(f'Vector Scaling: v = [{vec[0]}, {vec[1]}]', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    interact(plot_scaled_vector, 
             scale=FloatSlider(min=-3, max=3, step=0.1, value=1.0, 
                              description='Scale factor:', style={'description_width': 'initial'}))


def norm_explorer_widget():
    """
    Interactive widget to explore different p-norms and their unit balls.
    """
    def plot_p_norm(p):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        theta = np.linspace(0, 2*np.pi, 1000)
        
        if p == 0:
            # L0 "norm"
            x = np.array([1, -1, 0, 0])
            y = np.array([0, 0, 1, -1])
            ax.scatter(x, y, s=200, c='blue', zorder=5)
            label = r'$L_0$ "norm" (non-convex)'
        elif p >= 100:  # Approximate infinity
            # L-infinity norm
            x = np.array([-1, 1, 1, -1, -1])
            y = np.array([-1, -1, 1, 1, -1])
            ax.plot(x, y, linewidth=3, color='blue')
            ax.fill(x, y, alpha=0.3, color='blue')
            label = r'$L_\infty$ norm (max norm)'
        else:
            # General Lp norm
            x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
            y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
            ax.plot(x, y, linewidth=3, color='blue')
            ax.fill(x, y, alpha=0.3, color='blue')
            label = f'$L_{{{p:.1f}}}$ norm'
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title(f'Unit Ball: {label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add interpretation text
        if p < 1:
            interpretation = "p < 1: Non-convex, promotes sparsity"
        elif p == 1:
            interpretation = "p = 1: Manhattan/Taxicab norm, used in LASSO"
        elif p == 2:
            interpretation = "p = 2: Euclidean norm, most common in ML"
        elif p > 2 and p < 100:
            interpretation = f"p = {p:.1f}: Intermediate between L2 and L∞"
        else:
            interpretation = "p = ∞: Max norm, measures largest component"
        
        ax.text(0.5, -1.35, interpretation, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    interact(plot_p_norm,
             p=FloatSlider(min=0.5, max=10, step=0.1, value=2.0,
                          description='p value:', style={'description_width': 'initial'}))


def projection_widget(vector_x=[3, 2], vector_w=[2, 1]):
    """
    Interactive widget to explore vector projection.
    
    Parameters:
    -----------
    vector_x : list
        Vector to project
    vector_w : list
        Vector to project onto
    """
    def plot_projection(x1, x2, w1, w2):
        x = np.array([x1, x2])
        w = np.array([w1, w2])
        
        # Calculate projection
        proj = (np.dot(x, w) / np.dot(w, w)) * w
        perp = x - proj
        
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Plot vectors
        ax.arrow(0, 0, w[0], w[1], head_width=0.2, head_length=0.3,
                fc='blue', ec='blue', linewidth=2.5, label='Target (w)', zorder=3)
        ax.arrow(0, 0, x[0], x[1], head_width=0.2, head_length=0.3,
                fc='red', ec='red', linewidth=2.5, label='Original (x)', zorder=3)
        ax.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.3,
                fc='green', ec='green', linewidth=2, linestyle='--',
                label='Projection', zorder=2)
        ax.arrow(proj[0], proj[1], perp[0], perp[1], head_width=0.15, head_length=0.25,
                fc='orange', ec='orange', linewidth=2, linestyle=':',
                label='Perpendicular', zorder=2)
        
        # Draw perpendicular indicator
        if np.linalg.norm(perp) > 0.1:
            perp_indicator_size = 0.3
            perp_norm = perp / np.linalg.norm(perp)
            w_norm = w / np.linalg.norm(w)
            corner = proj + perp_indicator_size * (perp_norm + w_norm)
            square = np.array([proj, proj + perp_indicator_size * w_norm,
                             corner, proj + perp_indicator_size * perp_norm, proj])
            ax.plot(square[:, 0], square[:, 1], 'k-', linewidth=1, alpha=0.5)
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        
        max_val = max(abs(x[0]), abs(x[1]), abs(w[0]), abs(w[1])) + 1
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_title('Vector Projection Explorer', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Display projection formula and values
        proj_length = np.linalg.norm(proj)
        x_length = np.linalg.norm(x)
        angle = np.arccos(np.dot(x, w) / (np.linalg.norm(x) * np.linalg.norm(w))) * 180 / np.pi
        
        info_text = f'||proj|| = {proj_length:.2f}\n||x|| = {x_length:.2f}\nAngle = {angle:.1f}°'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    interact(plot_projection,
             x1=FloatSlider(min=-5, max=5, step=0.5, value=vector_x[0], description='x₁:'),
             x2=FloatSlider(min=-5, max=5, step=0.5, value=vector_x[1], description='x₂:'),
             w1=FloatSlider(min=-5, max=5, step=0.5, value=vector_w[0], description='w₁:'),
             w2=FloatSlider(min=-5, max=5, step=0.5, value=vector_w[1], description='w₂:'))


def svd_rank_widget(image_array):
    """
    Interactive widget to explore SVD compression with rank slider.
    
    Parameters:
    -----------
    image_array : array-like
        Grayscale image as 2D array
    """
    # Perform SVD once
    U, s, Vt = np.linalg.svd(image_array, full_matrices=False)
    max_rank = min(image_array.shape)
    
    def plot_svd_rank(rank):
        # Reconstruct with truncated SVD
        reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        
        # Calculate metrics
        original_size = image_array.size
        compressed_size = rank * (U.shape[0] + Vt.shape[1] + 1)
        compression_ratio = original_size / compressed_size
        error = np.linalg.norm(image_array - reconstructed, 'fro') / np.linalg.norm(image_array, 'fro')
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image_array, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Reconstructed
        axes[1].imshow(reconstructed, cmap='gray')
        axes[1].set_title(f'Rank {rank} Reconstruction', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Singular values
        axes[2].semilogy(s, 'b-', linewidth=2, label='All singular values')
        axes[2].semilogy(range(rank), s[:rank], 'ro', markersize=8, label=f'Used (rank {rank})')
        axes[2].axvline(x=rank-1, color='r', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Index', fontsize=11)
        axes[2].set_ylabel('Singular Value (log scale)', fontsize=11)
        axes[2].set_title('Singular Value Spectrum', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Add metrics text
        metrics_text = f'Compression: {compression_ratio:.1f}×\nRelative Error: {error:.2%}\nData retained: {100*rank/max_rank:.1f}%'
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.show()
    
    interact(plot_svd_rank,
             rank=IntSlider(min=1, max=max_rank, step=1, value=min(20, max_rank),
                           description='Rank:', style={'description_width': 'initial'}))


def condition_number_widget():
    """
    Interactive widget to explore matrix conditioning and scaling effects.
    """
    def plot_conditioning(scale_ratio, noise_level):
        # Create an ill-conditioned matrix with controllable condition number
        # Using a diagonal matrix with varying scales
        A = np.array([[1.0, 0.5],
                     [0.5, 1.0/scale_ratio]])
        
        b = np.array([1.0, 1.0])
        
        # Solve the system
        x_true = np.linalg.solve(A, b)
        
        # Add noise to b
        b_noisy = b + noise_level * np.random.randn(2)
        x_noisy = np.linalg.solve(A, b_noisy)
        
        # Calculate condition number
        cond_num = np.linalg.cond(A)
        
        # Relative errors
        b_error = np.linalg.norm(b_noisy - b) / np.linalg.norm(b)
        x_error = np.linalg.norm(x_noisy - x_true) / np.linalg.norm(x_true)
        error_amplification = x_error / b_error if b_error > 1e-10 else 0
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Matrix heatmap
        im = axes[0].imshow(A, cmap='RdBu_r', aspect='auto')
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, f'{A[i, j]:.3f}', ha="center", va="center", 
                           color="black", fontsize=14, fontweight='bold')
        axes[0].set_title(f'Matrix A (κ = {cond_num:.1f})', fontsize=12, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        plt.colorbar(im, ax=axes[0])
        
        # Plot 2: Solution comparison
        axes[1].scatter([0], [0], s=200, c='blue', marker='o', label='Origin', zorder=3)
        axes[1].scatter(x_true[0], x_true[1], s=200, c='green', marker='s', 
                       label='True solution', zorder=3)
        axes[1].scatter(x_noisy[0], x_noisy[1], s=200, c='red', marker='^',
                       label='Noisy solution', zorder=3)
        axes[1].arrow(x_true[0], x_true[1], x_noisy[0]-x_true[0], x_noisy[1]-x_true[1],
                     head_width=0.1, head_length=0.1, fc='orange', ec='orange', 
                     linewidth=2, linestyle='--', alpha=0.7)
        
        axes[1].set_xlabel('x₁', fontsize=12)
        axes[1].set_ylabel('x₂', fontsize=12)
        axes[1].set_title('Solution Space', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        
        # Add metrics
        metrics_text = (f'Condition Number: {cond_num:.2f}\n'
                       f'Input Error: {b_error:.2%}\n'
                       f'Output Error: {x_error:.2%}\n'
                       f'Error Amplification: {error_amplification:.2f}×')
        
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
    
    interact(plot_conditioning,
             scale_ratio=FloatSlider(min=1, max=100, step=1, value=10,
                                    description='Scale Ratio:', 
                                    style={'description_width': 'initial'}),
             noise_level=FloatSlider(min=0, max=0.5, step=0.01, value=0.1,
                                    description='Noise Level:',
                                    style={'description_width': 'initial'}))
