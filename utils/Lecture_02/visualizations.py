"""
Visualization functions for Linear Algebra concepts.

This module provides plotting and animation utilities for vectors, matrices,
norms, projections, eigendecompositions, and SVD.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import io
import base64
from IPython.display import HTML, Image


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def plot_vector_2d(vectors, labels=None, colors=None, title="2D Vectors", 
                   xlim=(-5, 5), ylim=(-5, 5), figsize=(8, 8), grid=True):
    """
    Plot 2D vectors as arrows from origin.
    
    Parameters:
    -----------
    vectors : list of array-like
        List of 2D vectors to plot
    labels : list of str, optional
        Labels for each vector
    colors : list of str, optional
        Colors for each vector
    title : str
        Plot title
    xlim, ylim : tuple
        Axis limits
    figsize : tuple
        Figure size
    grid : bool
        Show grid
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
    if labels is None:
        labels = [f"v{i+1}" for i in range(len(vectors))]
    
    for i, vec in enumerate(vectors):
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.2, head_length=0.3,
                fc=colors[i], ec=colors[i], linewidth=2, label=labels[i])
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_aspect('equal')
    if grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_vector_3d(vectors, labels=None, colors=None, title="3D Vectors",
                   xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5), figsize=(10, 8)):
    """
    Plot 3D vectors as arrows from origin.
    
    Parameters:
    -----------
    vectors : list of array-like
        List of 3D vectors to plot
    labels : list of str, optional
        Labels for each vector
    colors : list of str, optional
        Colors for each vector
    title : str
        Plot title
    xlim, ylim, zlim : tuple
        Axis limits
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(vectors)))
    if labels is None:
        labels = [f"v{i+1}" for i in range(len(vectors))]
    
    for i, vec in enumerate(vectors):
        arrow = Arrow3D([0, vec[0]], [0, vec[1]], [0, vec[2]],
                       mutation_scale=20, lw=2, arrowstyle="-|>",
                       color=colors[i], label=labels[i])
        ax.add_artist(arrow)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_zlabel('x₃', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig, ax


def plot_matrix_heatmap(matrix, title="Matrix Heatmap", cmap='RdBu_r',
                        figsize=(8, 6), annot=True, fmt='.2f'):
    """
    Plot matrix as a heatmap.
    
    Parameters:
    -----------
    matrix : array-like
        Matrix to visualize
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    annot : bool
        Annotate cells with values
    fmt : str
        Format string for annotations
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)
    
    # Annotate cells
    if annot and matrix.size <= 100:  # Only annotate if not too large
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, format(matrix[i, j], fmt),
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    
    plt.tight_layout()
    return fig, ax


def plot_norm_unit_balls(p_values=[0.5, 1, 2, np.inf], figsize=(12, 10)):
    """
    Plot unit balls for different p-norms.
    
    Parameters:
    -----------
    p_values : list
        List of p values for Lp norms
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    n_plots = len(p_values)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    theta = np.linspace(0, 2*np.pi, 1000)
    
    for idx, p in enumerate(p_values):
        ax = axes[idx]
        
        if p == np.inf:
            # L-infinity norm: square
            x = np.array([-1, 1, 1, -1, -1])
            y = np.array([-1, -1, 1, 1, -1])
            label = r'$L_\infty$ norm'
        elif p == 0:
            # L0 "norm": points at unit distance
            x = np.array([1, -1, 0, 0])
            y = np.array([0, 0, 1, -1])
            ax.scatter(x, y, s=100, c='blue')
            label = r'$L_0$ "norm"'
        else:
            # General Lp norm
            x = np.sign(np.cos(theta)) * np.abs(np.cos(theta))**(2/p)
            y = np.sign(np.sin(theta)) * np.abs(np.sin(theta))**(2/p)
            label = f'$L_{p}$ norm'
        
        if p != 0:
            ax.plot(x, y, linewidth=2, label=label)
            ax.fill(x, y, alpha=0.3)
        
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x₁', fontsize=11)
        ax.set_ylabel('x₂', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Unit Balls for Different p-Norms', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig, axes


def animate_projection(vector, onto_vector, n_frames=60, figsize=(8, 8)):
    """
    Create animation showing vector projection.
    
    Parameters:
    -----------
    vector : array-like
        Vector to project (2D)
    onto_vector : array-like
        Vector to project onto (2D)
    n_frames : int
        Number of animation frames
    figsize : tuple
        Figure size
    
    Returns:
    --------
    HTML animation object
    """
    vector = np.array(vector)
    onto_vector = np.array(onto_vector)
    
    # Calculate projection
    proj = (np.dot(vector, onto_vector) / np.dot(onto_vector, onto_vector)) * onto_vector
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up plot limits
    all_points = np.array([vector, onto_vector, proj, [0, 0]])
    margin = 1.5
    xlim = [all_points[:, 0].min() - margin, all_points[:, 0].max() + margin]
    ylim = [all_points[:, 1].min() - margin, all_points[:, 1].max() + margin]
    
    def init():
        ax.clear()
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        return []
    
    def animate(frame):
        ax.clear()
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Always show the target vector
        ax.arrow(0, 0, onto_vector[0], onto_vector[1], head_width=0.2, head_length=0.3,
                fc='blue', ec='blue', linewidth=2, label='Target vector (w)')
        
        # Always show original vector
        ax.arrow(0, 0, vector[0], vector[1], head_width=0.2, head_length=0.3,
                fc='red', ec='red', linewidth=2, label='Original vector (x)')
        
        # Animate projection growing
        if frame < n_frames // 2:
            t = frame / (n_frames // 2)
            current_proj = proj * t
            ax.arrow(0, 0, current_proj[0], current_proj[1], head_width=0.2, head_length=0.3,
                    fc='green', ec='green', linewidth=2, linestyle='--', 
                    label='Projection (proj_w(x))')
        else:
            # Show projection and perpendicular component
            ax.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.3,
                    fc='green', ec='green', linewidth=2, linestyle='--',
                    label='Projection (proj_w(x))')
            
            # Perpendicular component
            perp = vector - proj
            ax.arrow(proj[0], proj[1], perp[0], perp[1], head_width=0.2, head_length=0.3,
                    fc='orange', ec='orange', linewidth=2, linestyle=':',
                    label='Perpendicular component')
        
        ax.set_title('Vector Projection Animation', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, 
                        interval=50, blit=True, repeat=True)
    plt.close()
    
    return HTML(anim.to_jshtml())


def animate_eigen_transform(matrix, n_frames=120, figsize=(10, 8)):
    """
    Animate linear transformation showing eigenvector behavior.
    
    Parameters:
    -----------
    matrix : array-like (2x2)
        Transformation matrix
    n_frames : int
        Number of animation frames
    figsize : tuple
        Figure size
    
    Returns:
    --------
    HTML animation object
    """
    A = np.array(matrix)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Create a grid of vectors
    n_vectors = 8
    angles = np.linspace(0, 2*np.pi, n_vectors, endpoint=False)
    original_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    def init():
        for ax in [ax1, ax2]:
            ax.clear()
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        ax1.set_title('Original Vectors', fontsize=12, fontweight='bold')
        ax2.set_title('Transformed Vectors', fontsize=12, fontweight='bold')
        return []
    
    def animate(frame):
        for ax in [ax1, ax2]:
            ax.clear()
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        t = frame / n_frames
        
        # Plot original vectors
        for vec in original_vectors:
            ax1.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.15,
                     fc='blue', ec='blue', linewidth=1.5, alpha=0.6)
        
        # Plot eigenvectors in original space
        for i, eigvec in enumerate(eigenvectors.T):
            eigvec_real = np.real(eigvec)
            ax1.arrow(0, 0, eigvec_real[0], eigvec_real[1], head_width=0.15, head_length=0.2,
                     fc='red', ec='red', linewidth=2, label=f'Eigenvector {i+1}' if frame == 0 else '')
        
        ax1.set_title('Original Vectors', fontsize=12, fontweight='bold')
        
        # Animate transformation
        transformed = original_vectors @ A.T * t + original_vectors * (1 - t)
        
        for vec in transformed:
            ax2.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.15,
                     fc='green', ec='green', linewidth=1.5, alpha=0.6)
        
        # Plot transformed eigenvectors (scaled by eigenvalue)
        for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigvec_real = np.real(eigvec)
            transformed_eig = eigvec_real * np.real(eigval) * t + eigvec_real * (1 - t)
            ax2.arrow(0, 0, transformed_eig[0], transformed_eig[1], 
                     head_width=0.15, head_length=0.2,
                     fc='red', ec='red', linewidth=2)
        
        ax2.set_title(f'Transformed Vectors (t={t:.2f})', fontsize=12, fontweight='bold')
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=50, blit=True, repeat=True)
    plt.close()
    
    return HTML(anim.to_jshtml())


def plot_svd_compression(image_array, ranks=[1, 5, 10, 20, 50], figsize=(15, 10)):
    """
    Visualize image compression using truncated SVD.
    
    Parameters:
    -----------
    image_array : array-like
        Grayscale image as 2D array
    ranks : list of int
        Ranks to use for truncated SVD
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # Perform SVD
    U, s, Vt = np.linalg.svd(image_array, full_matrices=False)
    
    n_plots = len(ranks) + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image_array, cmap='gray')
    axes[0].set_title(f'Original\n({image_array.shape[0]}×{image_array.shape[1]} = {image_array.size} values)',
                     fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Compressed versions
    for idx, rank in enumerate(ranks, start=1):
        # Reconstruct with truncated SVD
        reconstructed = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        
        # Calculate compression ratio
        original_size = image_array.size
        compressed_size = rank * (U.shape[0] + Vt.shape[1] + 1)
        compression_ratio = original_size / compressed_size
        
        # Calculate error
        error = np.linalg.norm(image_array - reconstructed, 'fro') / np.linalg.norm(image_array, 'fro')
        
        axes[idx].imshow(reconstructed, cmap='gray')
        axes[idx].set_title(f'Rank {rank}\nCompression: {compression_ratio:.1f}x, Error: {error:.2%}',
                          fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('SVD Image Compression at Different Ranks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig, axes
