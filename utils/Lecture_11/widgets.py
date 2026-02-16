"""
Interactive widgets for exploring CNN concepts.

Provides ipywidgets-based interactive explorers.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def conv2d_widget():
    """
    Interactive widget for exploring 2D convolution parameters.
    
    Allows user to adjust:
    - Kernel type (from common_kernels)
    - Kernel size
    - Stride
    - Padding
    
    Shows live output of convolution operation.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return
    
    from ..Lecture_11.convolution import conv2d, common_kernels
    
    # Generate sample image
    image = np.random.rand(32, 32)
    kernels_dict = common_kernels()
    
    def update(kernel_name='edge_horizontal', stride=1, padding=0):
        kernel = kernels_dict[kernel_name]
        result = conv2d(image, kernel, padding=padding, stride=stride)
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(kernel, cmap='RdBu_r')
        axes[1].set_title(f'Kernel: {kernel_name}', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(result, cmap='gray')
        axes[2].set_title(f'Output {result.shape}', fontsize=11, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(update,
            kernel_name=widgets.Dropdown(options=list(kernels_dict.keys()),
                                        description='Kernel:'),
            stride=widgets.IntSlider(min=1, max=4, value=1, description='Stride:'),
            padding=widgets.IntSlider(min=0, max=3, value=0, description='Padding:'))


def pooling_widget():
    """
    Interactive widget for exploring pooling operations.
    
    Allows user to adjust:
    - Pool size
    - Stride
    - Mode (max vs avg)
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return
    
    from ..Lecture_11.pooling import max_pool2d, avg_pool2d
    
    # Generate sample feature map
    feature_map = np.random.rand(16, 16)
    
    def update(pool_size=2, stride=2, mode='max'):
        if mode == 'max':
            result = max_pool2d(feature_map, pool_size, stride)
        else:
            result = avg_pool2d(feature_map, pool_size, stride)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(feature_map, cmap='viridis')
        axes[0].set_title(f'Input {feature_map.shape}', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(result, cmap='viridis')
        axes[1].set_title(f'{mode.upper()} Pool Output {result.shape}',
                         fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(update,
            pool_size=widgets.IntSlider(min=2, max=4, value=2, description='Pool Size:'),
            stride=widgets.IntSlider(min=1, max=4, value=2, description='Stride:'),
            mode=widgets.Dropdown(options=['max', 'avg'], description='Mode:'))


def augmentation_widget():
    """
    Interactive widget for exploring data augmentation.
    
    Allows user to adjust:
    - Horizontal flip
    - Vertical flip
    - Rotation angle
    - Noise level
    - Brightness
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return
    
    from ..Lecture_11.augmentation import augment_numpy
    
    # Generate sample image
    image = np.random.rand(64, 64)
    
    def update(flip_h=False, flip_v=False, rotate=0, noise=0.0, brightness=0.0):
        aug = augment_numpy(image, flip_h, flip_v, rotate, noise, brightness)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(aug, cmap='gray')
        axes[1].set_title('Augmented', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(update,
            flip_h=widgets.Checkbox(value=False, description='Flip Horizontal'),
            flip_v=widgets.Checkbox(value=False, description='Flip Vertical'),
            rotate=widgets.FloatSlider(min=-180, max=180, value=0, description='Rotate (deg):'),
            noise=widgets.FloatSlider(min=0, max=0.1, step=0.01, value=0, description='Noise:'),
            brightness=widgets.FloatSlider(min=-0.5, max=0.5, step=0.05, value=0, description='Brightness:'))


def architecture_explorer_widget():
    """
    Interactive widget for exploring CNN architectures.
    
    Allows user to:
    - Select architecture
    - Adjust depth/filters
    - See parameter count
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return
    
    print("Architecture explorer widget")
    print("Select architecture and see parameter counts")
    
    def update(arch='SimpleCNN', n_filters=32, n_layers=2):
        print(f"\n{arch} Configuration:")
        print(f"  Filters: {n_filters}")
        print(f"  Layers: {n_layers}")
        print(f"  Estimated parameters: ~{n_filters * n_layers * 1000:,}")
        print("\nSee notebook for detailed architecture visualization")
    
    interact(update,
            arch=widgets.Dropdown(options=['SimpleCNN', 'LeNet5', 'VGG', 'ResNet'],
                                 description='Architecture:'),
            n_filters=widgets.IntSlider(min=16, max=128, step=16, value=32,
                                       description='Filters:'),
            n_layers=widgets.IntSlider(min=2, max=8, value=2,
                                      description='Layers:'))


def training_widget():
    """
    Interactive widget for exploring training hyperparameters.
    
    Allows user to adjust:
    - Learning rate
    - Batch size
    - Epochs
    
    Shows simulated training curve.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return
    
    def update(lr=0.001, batch_size=32, epochs=10):
        # Simulate training curve
        np.random.seed(42)
        
        # Loss decreases with some noise
        train_loss = np.exp(-np.linspace(0, 2, epochs)) + np.random.rand(epochs) * 0.1
        val_loss = np.exp(-np.linspace(0, 1.8, epochs)) + np.random.rand(epochs) * 0.15
        
        # Accuracy increases
        train_acc = 1 - np.exp(-np.linspace(0, 2.5, epochs)) + np.random.rand(epochs) * 0.05
        val_acc = 1 - np.exp(-np.linspace(0, 2.2, epochs)) + np.random.rand(epochs) * 0.08
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(range(1, epochs+1), train_loss, 'b-', label='Train', linewidth=2)
        axes[0].plot(range(1, epochs+1), val_loss, 'r-', label='Val', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title(f'Loss (LR={lr}, BS={batch_size})', fontsize=11, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(range(1, epochs+1), train_acc, 'b-', label='Train', linewidth=2)
        axes[1].plot(range(1, epochs+1), val_acc, 'r-', label='Val', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title('Accuracy', fontsize=11, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    interact(update,
            lr=widgets.FloatLogSlider(min=-5, max=-2, step=0.5, value=-3,
                                     description='Learning Rate:', readout_format='.4f'),
            batch_size=widgets.Dropdown(options=[16, 32, 64, 128],
                                       value=32, description='Batch Size:'),
            epochs=widgets.IntSlider(min=5, max=50, value=10, description='Epochs:'))


if __name__ == "__main__":
    print("Widget utilities loaded successfully!")
    print("Note: Widgets require ipywidgets and Jupyter environment")
