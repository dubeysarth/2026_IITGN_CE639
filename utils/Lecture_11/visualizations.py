"""
Visualization utilities for CNNs.

Provides rich plotting functions for convolution, feature maps, architectures, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import Tuple, Dict, List, Optional


def plot_conv2d_animation(image: np.ndarray, kernel: np.ndarray,
                          padding: int = 0, stride: int = 1,
                          figsize: Tuple[int, int] = (14, 6),
                          interval: int = 200) -> animation.FuncAnimation:
    """
    Create animation of 2D convolution sliding window.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W)
    kernel : np.ndarray
        Convolution kernel, shape (k, k)
    padding : int
        Padding
    stride : int
        Stride
    figsize : Tuple[int, int]
        Figure size
    interval : int
        Milliseconds between frames
        
    Returns
    -------
    animation.FuncAnimation
        Matplotlib animation object
    """
    from ..Lecture_11.convolution import conv2d_step_by_step
    
    # Collect all steps
    steps = list(conv2d_step_by_step(image, kernel, padding, stride))
    
    # Setup figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Titles
    axes[0].set_title('Input Image + Sliding Window', fontsize=12, fontweight='bold')
    axes[1].set_title('Kernel', fontsize=12, fontweight='bold')
    axes[2].set_title('Output Feature Map', fontsize=12, fontweight='bold')
    
    # Initialize plots
    im0 = axes[0].imshow(image, cmap='viridis')
    rect = Rectangle((0, 0), kernel.shape[1], kernel.shape[0],
                     linewidth=3, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    
    im1 = axes[1].imshow(kernel, cmap='RdBu_r')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(steps[0]['output_so_far'], cmap='viridis')
    plt.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.axis('off')
    
    # Animation update function
    def update(frame):
        step = steps[frame]
        
        # Update rectangle position
        h_start, w_start = step['window_coords']
        rect.set_xy((w_start, h_start))
        
        # Update output
        im2.set_data(step['output_so_far'])
        
        # Update title with current value
        axes[2].set_title(f'Output (current value: {step["value"]:.2f})',
                         fontsize=12, fontweight='bold')
        
        return [rect, im2]
    
    anim = animation.FuncAnimation(fig, update, frames=len(steps),
                                  interval=interval, blit=True, repeat=True)
    
    plt.tight_layout()
    
    return anim


def plot_feature_maps(model, image: np.ndarray, layer_idx: int,
                     n_maps: int = 16, figsize: Tuple[int, int] = (15, 8)):
    """
    Visualize feature maps from a specific layer.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    image : np.ndarray
        Input image
    layer_idx : int
        Layer index to visualize
    n_maps : int
        Number of feature maps to show
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    import torch
    
    # This is a placeholder - full implementation requires hooking into model
    print("Feature map visualization requires model hooks")
    print("See notebook for complete implementation")
    
    return None, None


def plot_kernel_gallery(kernels_dict: Dict[str, np.ndarray],
                       image: np.ndarray,
                       figsize: Tuple[int, int] = (16, 10)) -> Tuple:
    """
    Apply multiple kernels to an image and show results.
    
    Parameters
    ----------
    kernels_dict : Dict[str, np.ndarray]
        Dictionary of named kernels
    image : np.ndarray
        Input image
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    from ..Lecture_11.convolution import conv2d
    
    n_kernels = len(kernels_dict)
    n_cols = 4
    n_rows = (n_kernels + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (name, kernel) in enumerate(kernels_dict.items()):
        # Apply convolution
        result = conv2d(image, kernel)
        
        # Plot
        axes[i].imshow(result, cmap='gray')
        axes[i].set_title(name.replace('_', ' ').title(), fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_kernels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig, axes


def plot_pooling_comparison(feature_map: np.ndarray,
                           pool_sizes: List[int] = [2, 3, 4],
                           figsize: Tuple[int, int] = (15, 5)) -> Tuple:
    """
    Compare max vs average pooling with different pool sizes.
    
    Parameters
    ----------
    feature_map : np.ndarray
        Input feature map
    pool_sizes : List[int]
        Pool sizes to compare
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    from ..Lecture_11.pooling import max_pool2d, avg_pool2d
    
    n_sizes = len(pool_sizes)
    fig, axes = plt.subplots(2, n_sizes + 1, figsize=figsize)
    
    # Original
    axes[0, 0].imshow(feature_map, cmap='viridis')
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Pooling comparisons
    for i, pool_size in enumerate(pool_sizes):
        # Max pooling
        max_pooled = max_pool2d(feature_map, pool_size, stride=pool_size)
        axes[0, i+1].imshow(max_pooled, cmap='viridis')
        axes[0, i+1].set_title(f'Max Pool {pool_size}×{pool_size}\\n{max_pooled.shape}',
                              fontsize=10)
        axes[0, i+1].axis('off')
        
        # Average pooling
        avg_pooled = avg_pool2d(feature_map, pool_size, stride=pool_size)
        axes[1, i+1].imshow(avg_pooled, cmap='viridis')
        axes[1, i+1].set_title(f'Avg Pool {pool_size}×{pool_size}\\n{avg_pooled.shape}',
                              fontsize=10)
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    return fig, axes


def plot_architecture_diagram(arch_name: str,
                              figsize: Tuple[int, int] = (14, 6)) -> Tuple:
    """
    Plot schematic diagram of CNN architecture.
    
    Parameters
    ----------
    arch_name : str
        Architecture name: 'lenet', 'vgg', 'resnet'
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Placeholder - draw simple box diagram
    ax.text(0.5, 0.5, f'{arch_name.upper()} Architecture Diagram',
           ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.4, 'See notebook for detailed implementation',
           ha='center', va='center', fontsize=12, style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    return fig, ax


def plot_receptive_field(n_layers: int, kernel_size: int = 3,
                        figsize: Tuple[int, int] = (12, 6)) -> Tuple:
    """
    Visualize growing receptive field through CNN layers.
    
    Parameters
    ----------
    n_layers : int
        Number of convolutional layers
    kernel_size : int
        Kernel size
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    # Calculate receptive field size at each layer
    receptive_fields = [kernel_size]
    for i in range(1, n_layers):
        rf = receptive_fields[-1] + (kernel_size - 1)
        receptive_fields.append(rf)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    layers = range(1, n_layers + 1)
    ax.plot(layers, receptive_fields, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('Receptive Field Size', fontsize=12)
    ax.set_title(f'Receptive Field Growth (kernel={kernel_size}×{kernel_size})',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotate
    for i, (layer, rf) in enumerate(zip(layers, receptive_fields)):
        ax.annotate(f'{rf}×{rf}', (layer, rf), textcoords='offset points',
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    return fig, ax


def plot_parameter_comparison(image_size: int = 32, n_hidden: int = 128,
                              figsize: Tuple[int, int] = (10, 6)) -> Tuple:
    """
    Compare parameters: FC vs Conv layers.
    
    Parameters
    ----------
    image_size : int
        Input image size (square)
    n_hidden : int
        Number of hidden units (FC) or filters (Conv)
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    # Calculate parameters
    fc_params = (image_size * image_size * 3) * n_hidden
    conv_params = n_hidden * (3 * 3 * 3 + 1)  # 3x3 kernel, 3 channels, bias
    
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = ['Fully Connected', 'Convolutional']
    params = [fc_params, conv_params]
    colors = ['coral', 'steelblue']
    
    bars = ax.bar(methods, params, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title(f'Parameter Count Comparison\\n({image_size}×{image_size} input, {n_hidden} units/filters)',
                fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{param:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add reduction factor
    reduction = fc_params / conv_params
    ax.text(0.5, 0.95, f'{reduction:.1f}× fewer parameters with Conv!',
           transform=ax.transAxes, ha='center', va='top',
           fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    return fig, ax


def plot_feature_hierarchy(model, image: np.ndarray,
                          figsize: Tuple[int, int] = (15, 10)) -> Tuple:
    """
    Visualize feature hierarchy: early/mid/deep layers.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    image : np.ndarray
        Input image
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    # Placeholder
    print("Feature hierarchy visualization requires model hooks")
    print("See notebook for complete implementation")
    
    return None, None


def plot_augmentation_grid(image: np.ndarray, transforms: List,
                          figsize: Tuple[int, int] = (12, 8)) -> Tuple:
    """
    Show grid of different augmentations applied to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    transforms : List
        List of augmentation functions
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    from ..Lecture_11.augmentation import augmentation_gallery
    
    return augmentation_gallery(image, n_augmented=8, figsize=figsize)


def plot_confusion_matrix_cnn(y_true: np.ndarray, y_pred: np.ndarray,
                              classes: List[str],
                              figsize: Tuple[int, int] = (8, 7)) -> Tuple:
    """
    Plot styled confusion matrix for CNN predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    classes : List[str]
        Class names
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, cm[i, j], ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black',
                          fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig, ax


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
