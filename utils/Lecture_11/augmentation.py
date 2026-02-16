"""
Data augmentation utilities for CNNs.

Provides both NumPy-based and PyTorch-based augmentation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from scipy.ndimage import rotate as scipy_rotate


def augment_numpy(image: np.ndarray, flip_horizontal: bool = False,
                  flip_vertical: bool = False, rotate_deg: float = 0,
                  noise_std: float = 0, brightness: float = 0) -> np.ndarray:
    """
    Apply augmentations to an image using NumPy.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C)
    flip_horizontal : bool
        Flip horizontally
    flip_vertical : bool
        Flip vertically
    rotate_deg : float
        Rotation angle in degrees
    noise_std : float
        Standard deviation of Gaussian noise
    brightness : float
        Brightness adjustment (-1 to 1)
        
    Returns
    -------
    np.ndarray
        Augmented image
    """
    aug = image.copy()
    
    # Flips
    if flip_horizontal:
        aug = np.fliplr(aug)
    if flip_vertical:
        aug = np.flipud(aug)
    
    # Rotation
    if rotate_deg != 0:
        aug = scipy_rotate(aug, rotate_deg, reshape=False, mode='nearest')
    
    # Noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, aug.shape)
        aug = aug + noise
    
    # Brightness
    if brightness != 0:
        aug = aug + brightness
    
    # Clip to valid range
    aug = np.clip(aug, 0, 1)
    
    return aug


def augmentation_gallery(image: np.ndarray, n_augmented: int = 8,
                         figsize: Tuple[int, int] = (15, 8)) -> Tuple:
    """
    Create a gallery of augmented versions of an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C)
    n_augmented : int
        Number of augmented versions to generate
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray, List[np.ndarray]]
        Figure, axes, and list of augmented images
    """
    n_cols = 4
    n_rows = (n_augmented + n_cols) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    augmented_images = []
    
    # Generate augmentations
    for i in range(1, n_augmented + 1):
        # Random augmentation parameters
        flip_h = np.random.rand() > 0.5
        flip_v = np.random.rand() > 0.7
        rotate = np.random.uniform(-30, 30)
        noise = np.random.uniform(0, 0.05)
        brightness = np.random.uniform(-0.2, 0.2)
        
        aug = augment_numpy(image, flip_h, flip_v, rotate, noise, brightness)
        augmented_images.append(aug)
        
        axes[i].imshow(aug, cmap='gray' if aug.ndim == 2 else None)
        
        # Create title with augmentation params
        params = []
        if flip_h:
            params.append('FlipH')
        if flip_v:
            params.append('FlipV')
        if abs(rotate) > 1:
            params.append(f'Rot{rotate:.0f}°')
        if noise > 0.01:
            params.append(f'Noise')
        if abs(brightness) > 0.05:
            params.append(f'Bright')
        
        title = ', '.join(params) if params else 'No aug'
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_augmented + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    return fig, axes, augmented_images


def get_augmentation_transform(mode: str = 'medium'):
    """
    Get torchvision augmentation transforms.
    
    Parameters
    ----------
    mode : str
        Augmentation intensity: 'light', 'medium', 'heavy'
        
    Returns
    -------
    torchvision.transforms.Compose
        Composed augmentation transform
        
    Examples
    --------
    >>> transform = get_augmentation_transform('medium')
    >>> # Use with PyTorch dataset
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision required for PyTorch transforms")
    
    if mode == 'light':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])
    elif mode == 'medium':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                  saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])
    elif mode == 'heavy':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                  saturation=0.3, hue=0.2),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compare_with_without_augmentation(model_class, dataset, epochs: int = 10,
                                      device: str = 'cpu') -> Dict:
    """
    Train two models: one with augmentation, one without.
    
    Parameters
    ----------
    model_class : class
        Model class to instantiate
    dataset : tuple
        (train_data, val_data) tuple
    epochs : int
        Number of epochs
    device : str
        Device
        
    Returns
    -------
    Dict
        Results with keys:
        - 'with_aug': history with augmentation
        - 'without_aug': history without augmentation
    """
    # This is a placeholder - actual implementation would require
    # full dataset setup with/without augmentation
    print("Comparison function requires full dataset setup")
    print("See notebook for complete implementation")
    
    return {
        'with_aug': None,
        'without_aug': None
    }


if __name__ == "__main__":
    print("Testing augmentation.py...")
    
    # Create test image
    image = np.random.rand(64, 64)
    
    # Test single augmentation
    aug = augment_numpy(image, flip_horizontal=True, rotate_deg=15,
                       noise_std=0.02, brightness=0.1)
    print(f"Augmented image shape: {aug.shape}")
    
    print("✓ Augmentation module loaded successfully!")
