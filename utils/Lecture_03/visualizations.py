"""
Visualization functions for Probability & Statistics concepts.

This module provides plotting and animation utilities for distributions,
sampling, covariance, and the Central Limit Theorem.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')


def plot_pmf(values, probabilities, title="Probability Mass Function", 
             xlabel="Value", ylabel="Probability", figsize=(10, 6)):
    """
    Plot a Probability Mass Function (discrete distribution).
    
    Parameters:
    -----------
    values : array-like
        Discrete values
    probabilities : array-like
        Probabilities for each value
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.stem(values, probabilities, basefmt=' ', linefmt='C0-', markerfmt='C0o')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_pdf(x, pdf_values, title="Probability Density Function",
             xlabel="Value", ylabel="Density", fill=True, figsize=(10, 6)):
    """
    Plot a Probability Density Function (continuous distribution).
    
    Parameters:
    -----------
    x : array-like
        x values
    pdf_values : array-like
        PDF values
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    fill : bool
        Fill area under curve
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, pdf_values, 'b-', linewidth=2, label='PDF')
    if fill:
        ax.fill_between(x, pdf_values, alpha=0.3)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_cdf(x, cdf_values, title="Cumulative Distribution Function",
             xlabel="Value", ylabel="Cumulative Probability", figsize=(10, 6)):
    """
    Plot a Cumulative Distribution Function.
    
    Parameters:
    -----------
    x : array-like
        x values
    cdf_values : array-like
        CDF values
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, cdf_values, 'r-', linewidth=2, label='CDF')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_distribution_comparison(x, distributions, labels, title="Distribution Comparison",
                                 xlabel="Value", ylabel="Density", figsize=(12, 6)):
    """
    Compare multiple distributions on the same plot.
    
    Parameters:
    -----------
    x : array-like
        x values
    distributions : list of array-like
        List of PDF values for each distribution
    labels : list of str
        Labels for each distribution
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(distributions)))
    
    for i, (dist, label) in enumerate(zip(distributions, labels)):
        ax.plot(x, dist, linewidth=2, label=label, color=colors[i])
        ax.fill_between(x, dist, alpha=0.2, color=colors[i])
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def animate_sampling(dist_func, n_frames=100, samples_per_frame=10, 
                     max_samples=1000, bins=30, **dist_params):
    """
    Animate sampling from a distribution and convergence of histogram to PDF.
    
    Parameters:
    -----------
    dist_func : callable
        Function to generate samples (e.g., np.random.normal)
    n_frames : int
        Number of animation frames
    samples_per_frame : int
        Samples to add per frame
    max_samples : int
        Maximum total samples
    bins : int
        Number of histogram bins
    **dist_params : dict
        Parameters for distribution function
    
    Returns:
    --------
    HTML animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    all_samples = []
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Generate new samples
        new_samples = dist_func(size=samples_per_frame, **dist_params)
        all_samples.extend(new_samples)
        
        n_samples = len(all_samples)
        
        # Plot 1: Histogram
        ax1.hist(all_samples, bins=bins, density=True, alpha=0.7, 
                color='blue', edgecolor='black')
        ax1.set_xlabel('Value', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'Histogram (n={n_samples})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample evolution
        ax2.plot(all_samples, 'o', markersize=3, alpha=0.5)
        ax2.set_xlabel('Sample Index', fontsize=11)
        ax2.set_ylabel('Value', fontsize=11)
        ax2.set_title('Sample Values Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=100, blit=True, repeat=False)
    plt.close()
    
    return HTML(anim.to_jshtml())


def plot_covariance_matrix(cov_matrix, labels=None, title="Covariance Matrix",
                           figsize=(8, 6), annot=True):
    """
    Visualize a covariance matrix as a heatmap.
    
    Parameters:
    -----------
    cov_matrix : array-like
        Covariance matrix
    labels : list of str, optional
        Variable labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    annot : bool
        Annotate cells with values
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Covariance', rotation=270, labelpad=20)
    
    # Annotations
    if annot:
        for i in range(cov_matrix.shape[0]):
            for j in range(cov_matrix.shape[1]):
                text = ax.text(j, i, f'{cov_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    # Labels
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_correlation_scatter(x, y, title="Correlation Scatter Plot",
                             xlabel="X", ylabel="Y", show_line=True, figsize=(10, 8)):
    """
    Scatter plot with correlation coefficient and regression line.
    
    Parameters:
    -----------
    x, y : array-like
        Data arrays
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    show_line : bool
        Show regression line
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Calculate correlation
    correlation = np.corrcoef(x, y)[0, 1]
    
    # Regression line
    if show_line:
        coeffs = np.polyfit(x, y, 1)
        line = np.poly1d(coeffs)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, line(x_line), 'r--', linewidth=2, label=f'Regression line')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    if show_line:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


def animate_clt(dist_func, sample_sizes=[1, 5, 10, 30, 50, 100],
               n_experiments=1000, bins=30, **dist_params):
    """
    Animate the Central Limit Theorem - show sample means converging to normal.
    
    Parameters:
    -----------
    dist_func : callable
        Distribution to sample from
    sample_sizes : list of int
        Sample sizes to demonstrate
    n_experiments : int
        Number of experiments for each sample size
    bins : int
        Histogram bins
    **dist_params : dict
        Distribution parameters
    
    Returns:
    --------
    HTML animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        n = sample_sizes[frame % len(sample_sizes)]
        
        # Generate sample means
        sample_means = []
        for _ in range(n_experiments):
            samples = dist_func(size=n, **dist_params)
            sample_means.append(np.mean(samples))
        
        sample_means = np.array(sample_means)
        
        # Plot 1: Histogram of sample means
        ax1.hist(sample_means, bins=bins, density=True, alpha=0.7,
                color='blue', edgecolor='black', label='Sample means')
        
        # Overlay normal distribution
        mu = np.mean(sample_means)
        sigma = np.std(sample_means)
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        from scipy.stats import norm
        ax1.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        
        ax1.set_xlabel('Sample Mean', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title(f'Distribution of Sample Means (n={n})', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot
        from scipy.stats import probplot
        probplot(sample_means, dist="norm", plot=ax2)
        ax2.set_title(f'Q-Q Plot (n={n})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(sample_sizes) * 3,  # Repeat 3 times
                        interval=1000, blit=True, repeat=True)
    plt.close()
    
    return HTML(anim.to_jshtml())
