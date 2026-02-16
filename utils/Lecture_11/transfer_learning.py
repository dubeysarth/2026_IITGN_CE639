"""
Transfer learning utilities for CNNs.

Provides functions for loading pretrained models and fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


def load_pretrained_resnet(n_classes: int, freeze_backbone: bool = True,
                          pretrained: bool = True) -> nn.Module:
    """
    Load pretrained ResNet-18 and replace final layer.
    
    Parameters
    ----------
    n_classes : int
        Number of output classes for new task
    freeze_backbone : bool
        If True, freeze all layers except final FC
    pretrained : bool
        Load ImageNet pretrained weights
        
    Returns
    -------
    nn.Module
        Modified ResNet model
        
    Examples
    --------
    >>> model = load_pretrained_resnet(n_classes=2, freeze_backbone=True)
    >>> # Fine-tune on crack detection dataset
    """
    try:
        from torchvision import models
    except ImportError:
        raise ImportError("torchvision required for pretrained models")
    
    # Load pretrained ResNet-18
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_classes)
    
    return model


def freeze_layers(model: nn.Module, n_layers_to_freeze: int) -> nn.Module:
    """
    Freeze first n layers of a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    n_layers_to_freeze : int
        Number of layers to freeze from the start
        
    Returns
    -------
    nn.Module
        Model with frozen layers
    """
    layers = list(model.children())
    
    for i, layer in enumerate(layers):
        if i < n_layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    return model


def compare_scratch_vs_pretrained(dataset, n_classes: int, epochs: int = 10,
                                  device: str = 'cpu') -> Dict:
    """
    Compare training from scratch vs transfer learning.
    
    Parameters
    ----------
    dataset : tuple
        (train_loader, val_loader)
    n_classes : int
        Number of classes
    epochs : int
        Number of epochs
    device : str
        Device
        
    Returns
    -------
    Dict
        Results with keys:
        - 'scratch': history for model trained from scratch
        - 'pretrained': history for transfer learning model
    """
    # Placeholder - full implementation in notebook
    print("Comparison requires full training setup")
    print("See notebook for complete implementation")
    
    return {
        'scratch': None,
        'pretrained': None
    }


def visualize_frozen_vs_unfrozen(model: nn.Module, figsize=(12, 6)):
    """
    Visualize which layers are frozen vs trainable.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    figsize : tuple
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes
    """
    import matplotlib.pyplot as plt
    
    # Get layer info
    layer_names = []
    layer_frozen = []
    layer_params = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = list(module.parameters())
            if len(params) > 0:
                layer_names.append(name if name else 'root')
                layer_frozen.append(not params[0].requires_grad)
                layer_params.append(sum(p.numel() for p in params))
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['lightcoral' if frozen else 'lightgreen' 
              for frozen in layer_frozen]
    
    y_pos = range(len(layer_names))
    ax.barh(y_pos, layer_params, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel('Number of Parameters', fontsize=11)
    ax.set_title('Layer Parameters (Red=Frozen, Green=Trainable)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='Frozen'),
        Patch(facecolor='lightgreen', label='Trainable')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    return fig, ax


if __name__ == "__main__":
    print("Transfer learning utilities loaded successfully!")
    print("Note: Requires torchvision for pretrained models")
