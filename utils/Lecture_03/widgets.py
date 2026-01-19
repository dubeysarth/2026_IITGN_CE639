"""
Interactive widgets for exploring Probability & Statistics concepts.

This module provides ipywidgets-based interactive controls for
distribution exploration, sampling, covariance, and extreme values.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, fixed
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def distribution_explorer_widget(dist_type='normal'):
    """
    Interactive widget to explore distribution parameters.
    
    Parameters:
    -----------
    dist_type : str
        'normal', 'uniform', or 'gumbel'
    """
    def plot_distribution(mu, sigma):
        from .distributions import normal_pdf, normal_cdf
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Generate x values
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        
        # PDF
        pdf = normal_pdf(x, mu, sigma)
        ax1.plot(x, pdf, 'b-', linewidth=2)
        ax1.fill_between(x, pdf, alpha=0.3)
        ax1.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax1.axvline(mu - sigma, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label=f'μ ± σ')
        ax1.axvline(mu + sigma, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title(f'Normal PDF (μ={mu}, σ={sigma})', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        cdf = normal_cdf(x, mu, sigma)
        ax2.plot(x, cdf, 'r-', linewidth=2)
        ax2.axhline(0.5, color='b', linestyle='--', linewidth=1, alpha=0.5, label='Median')
        ax2.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title(f'Normal CDF (μ={mu}, σ={sigma})', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    if dist_type == 'normal':
        interact(plot_distribution,
                mu=FloatSlider(min=-10, max=10, step=0.5, value=0, 
                              description='Mean (μ):', style={'description_width': 'initial'}),
                sigma=FloatSlider(min=0.1, max=5, step=0.1, value=1,
                                 description='Std Dev (σ):', style={'description_width': 'initial'}))


def sampling_widget():
    """
    Interactive widget to demonstrate sampling and histogram convergence.
    """
    def plot_sampling(n_samples, mu, sigma):
        # Generate samples
        samples = np.random.normal(mu, sigma, n_samples)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        ax1.hist(samples, bins=30, density=True, alpha=0.7, color='blue', 
                edgecolor='black', label='Histogram')
        
        # Overlay true PDF
        x = np.linspace(samples.min(), samples.max(), 200)
        from .distributions import normal_pdf
        pdf = normal_pdf(x, mu, sigma)
        ax1.plot(x, pdf, 'r-', linewidth=2, label='True PDF')
        
        ax1.set_xlabel('Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'Sampling from N({mu}, {sigma}²) - n={n_samples}', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sample statistics
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        
        ax2.text(0.1, 0.9, 'Sample Statistics:', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.1, 0.75, f'Sample Mean: {sample_mean:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.65, f'True Mean: {mu:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.50, f'Sample Std: {sample_std:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.40, f'True Std: {sigma:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.25, f'Error in Mean: {abs(sample_mean - mu):.4f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.15, f'Error in Std: {abs(sample_std - sigma):.4f}', fontsize=12,
                transform=ax2.transAxes)
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(plot_sampling,
            n_samples=IntSlider(min=10, max=10000, step=10, value=100,
                               description='Sample Size:', style={'description_width': 'initial'}),
            mu=FloatSlider(min=-5, max=5, step=0.5, value=0,
                          description='Mean (μ):', style={'description_width': 'initial'}),
            sigma=FloatSlider(min=0.5, max=3, step=0.1, value=1,
                             description='Std Dev (σ):', style={'description_width': 'initial'}))


def covariance_widget():
    """
    Interactive widget to explore covariance and correlation.
    """
    def plot_covariance(correlation, n_samples):
        from .ce_examples import generate_correlated_data
        
        # Generate correlated data
        x, y = generate_correlated_data(n_samples=n_samples, 
                                        mean1=0, mean2=0,
                                        std1=1, std2=1,
                                        correlation=correlation)
        
        # Calculate actual correlation
        actual_corr = np.corrcoef(x, y)[0, 1]
        cov = np.cov(x, y)[0, 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot
        ax1.scatter(x, y, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Regression line
        coeffs = np.polyfit(x, y, 1)
        line = np.poly1d(coeffs)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, line(x_line), 'r--', linewidth=2, label='Regression line')
        
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_title(f'Correlated Data (ρ = {actual_corr:.3f})', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Covariance matrix
        cov_matrix = np.cov(x, y)
        im = ax2.imshow(cov_matrix, cmap='RdBu_r', aspect='auto')
        
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, f'{cov_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", 
                              fontsize=14, fontweight='bold')
        
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        ax2.set_xticklabels(['X', 'Y'])
        ax2.set_yticklabels(['X', 'Y'])
        ax2.set_title('Covariance Matrix', fontsize=13, fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        # Add text with statistics
        fig.text(0.5, 0.02, 
                f'Target ρ: {correlation:.2f} | Actual ρ: {actual_corr:.3f} | Cov(X,Y): {cov:.3f}',
                ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()
    
    interact(plot_covariance,
            correlation=FloatSlider(min=-0.95, max=0.95, step=0.05, value=0.7,
                                   description='Correlation (ρ):', 
                                   style={'description_width': 'initial'}),
            n_samples=IntSlider(min=50, max=500, step=50, value=200,
                               description='Sample Size:', 
                               style={'description_width': 'initial'}))


def extreme_value_widget():
    """
    Interactive widget to explore Generalized Extreme Value (GEV) distribution.
    """
    def plot_gev(xi, mu, sigma):
        from .distributions import gev_pdf, gev_cdf
        
        # Determine appropriate x range based on xi
        if xi > 0:
            x_min = mu - sigma/xi - 5
            x_max = mu + 20
        elif xi < 0:
            x_min = mu - 20
            x_max = mu - sigma/xi + 5
        else:  # Gumbel
            x_min = mu - 20
            x_max = mu + 20
        
        x = np.linspace(x_min, x_max, 500)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PDF
        pdf = gev_pdf(x, mu, sigma, xi)
        ax1.plot(x, pdf, 'b-', linewidth=2)
        ax1.fill_between(x, pdf, alpha=0.3)
        ax1.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        
        # Determine distribution type
        if abs(xi) < 0.01:
            dist_name = 'Gumbel (Type I)'
        elif xi > 0:
            dist_name = 'Fréchet (Type II)'
        else:
            dist_name = 'Weibull (Type III)'
        
        ax1.set_title(f'GEV PDF: {dist_name}\n(μ={mu}, σ={sigma}, ξ={xi})', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF
        cdf = gev_cdf(x, mu, sigma, xi)
        ax2.plot(x, cdf, 'r-', linewidth=2)
        ax2.axhline(0.5, color='b', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'μ = {mu}')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title(f'GEV CDF: {dist_name}', fontsize=12, fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation
        interpretation = {
            'Gumbel (Type I)': 'Exponential tail decay (e.g., floods, wind)',
            'Fréchet (Type II)': 'Heavy tail, no upper bound (e.g., insurance claims)',
            'Weibull (Type III)': 'Bounded above (e.g., material strength)'
        }
        
        fig.text(0.5, 0.02, interpretation[dist_name],
                ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        plt.show()
    
    interact(plot_gev,
            xi=FloatSlider(min=-0.5, max=0.5, step=0.05, value=0,
                          description='Shape (ξ):', 
                          style={'description_width': 'initial'}),
            mu=FloatSlider(min=0, max=100, step=5, value=50,
                          description='Location (μ):', 
                          style={'description_width': 'initial'}),
            sigma=FloatSlider(min=1, max=20, step=1, value=10,
                             description='Scale (σ):', 
                             style={'description_width': 'initial'}))


def moments_explorer_widget():
    """
    Interactive widget to explore effect of skewness and kurtosis.
    """
    def plot_moments(skewness_factor, kurtosis_factor, n_samples):
        # Generate data with specified moments
        # Start with normal
        data = np.random.randn(n_samples)
        
        # Add skewness (simple transformation)
        if skewness_factor != 0:
            data = np.sign(data) * np.abs(data) ** (1 + skewness_factor * 0.5)
        
        # Calculate moments
        from .distributions import compute_moments
        mean, var, skew, kurt = compute_moments(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(data, bins=40, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')
        ax1.axvline(mean - np.sqrt(var), color='g', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.axvline(mean + np.sqrt(var), color='g', linestyle=':', linewidth=1.5, alpha=0.7, 
                   label=f'±1 Std Dev')
        ax1.set_xlabel('Value', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Moments display
        ax2.text(0.1, 0.85, 'Statistical Moments:', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        ax2.text(0.1, 0.70, f'Mean (μ): {mean:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.60, f'Variance (σ²): {var:.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.50, f'Std Dev (σ): {np.sqrt(var):.3f}', fontsize=12,
                transform=ax2.transAxes)
        ax2.text(0.1, 0.40, f'Skewness: {skew:.3f}', fontsize=12,
                transform=ax2.transAxes, 
                color='red' if abs(skew) > 0.5 else 'black')
        ax2.text(0.1, 0.30, f'Kurtosis: {kurt:.3f}', fontsize=12,
                transform=ax2.transAxes,
                color='red' if abs(kurt) > 1 else 'black')
        
        # Interpretation
        ax2.text(0.1, 0.15, 'Interpretation:', fontsize=12, fontweight='bold',
                transform=ax2.transAxes)
        
        if skew > 0.5:
            skew_text = 'Right-skewed (long right tail)'
        elif skew < -0.5:
            skew_text = 'Left-skewed (long left tail)'
        else:
            skew_text = 'Approximately symmetric'
        
        if kurt > 1:
            kurt_text = 'Heavy tails (outliers likely)'
        elif kurt < -1:
            kurt_text = 'Light tails (few outliers)'
        else:
            kurt_text = 'Normal-like tails'
        
        ax2.text(0.1, 0.08, skew_text, fontsize=11, transform=ax2.transAxes)
        ax2.text(0.1, 0.02, kurt_text, fontsize=11, transform=ax2.transAxes)
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(plot_moments,
            skewness_factor=FloatSlider(min=-1, max=1, step=0.1, value=0,
                                       description='Skewness:', 
                                       style={'description_width': 'initial'}),
            kurtosis_factor=FloatSlider(min=-1, max=1, step=0.1, value=0,
                                       description='Kurtosis:', 
                                       style={'description_width': 'initial'}),
            n_samples=IntSlider(min=100, max=5000, step=100, value=1000,
                               description='Sample Size:', 
                               style={'description_width': 'initial'}))
