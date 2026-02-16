"""
CNN architectures using PyTorch.

Provides classic and modern CNN architectures for teaching purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class SimpleCNN(nn.Module):
    """
    Minimal 2-layer CNN for teaching basic concepts.
    
    Architecture:
    - Conv1: 3x3, 16 filters
    - ReLU + MaxPool
    - Conv2: 3x3, 32 filters
    - ReLU + MaxPool
    - Flatten
    - FC1: 128 units
    - FC2: n_classes units
    
    Parameters
    ----------
    n_classes : int
        Number of output classes
    input_channels : int
        Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, n_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Assuming 32x32 input → after 2 pools: 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture (LeCun et al., 1998).
    
    Original architecture for MNIST digit recognition.
    
    Architecture:
    - Conv1: 5x5, 6 filters
    - AvgPool
    - Conv2: 5x5, 16 filters
    - AvgPool
    - Conv3: 5x5, 120 filters
    - FC1: 84 units
    - FC2: n_classes units
    
    Parameters
    ----------
    n_classes : int
        Number of output classes (default: 10 for MNIST)
    input_channels : int
        Number of input channels (default: 1 for grayscale)
    """
    
    def __init__(self, n_classes: int = 10, input_channels: int = 1):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        
        self.pool = nn.AvgPool2d(2, 2)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, n_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.tanh(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(torch.tanh(self.conv2(x)))
        
        # Conv3 (no pooling)
        x = torch.tanh(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        
        return x


def make_vgg_block(in_channels: int, out_channels: int, 
                   n_convs: int = 2) -> nn.Sequential:
    """
    Create a VGG-style block: n_convs × (Conv3x3 + ReLU) + MaxPool.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_convs : int
        Number of conv layers in the block
        
    Returns
    -------
    nn.Sequential
        VGG block
        
    Examples
    --------
    >>> block = make_vgg_block(3, 64, n_convs=2)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> y = block(x)
    >>> y.shape
    torch.Size([1, 64, 112, 112])
    """
    layers = []
    
    for i in range(n_convs):
        layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                               out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
    
    layers.append(nn.MaxPool2d(2, 2))
    
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """
    Single residual block with skip connection.
    
    Implements: y = F(x) + x
    where F(x) is Conv-ReLU-Conv
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int
        Stride for first conv (for downsampling)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Skip connection
        out = F.relu(out)
        
        return out


class SimpleResNet(nn.Module):
    """
    Simplified ResNet for teaching purposes.
    
    Architecture:
    - Conv1: 7x7, 64 filters
    - MaxPool
    - ResBlock × 2 (64 filters)
    - ResBlock × 2 (128 filters, downsample)
    - ResBlock × 2 (256 filters, downsample)
    - Global Average Pool
    - FC
    
    Parameters
    ----------
    n_classes : int
        Number of output classes
    input_channels : int
        Number of input channels
    """
    
    def __init__(self, n_classes: int = 10, input_channels: int = 3):
        super(SimpleResNet, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)
    
    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        
        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc(x)
        
        return x


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count total and trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
        
    Returns
    -------
    Dict[str, int]
        Dictionary with 'total' and 'trainable' parameter counts
        
    Examples
    --------
    >>> model = LeNet5(n_classes=10)
    >>> params = count_parameters(model)
    >>> print(f"Total: {params['total']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable
    }


def model_summary(model: nn.Module, input_shape: Tuple[int, ...],
                  device: str = 'cpu') -> str:
    """
    Generate a pretty-printed model summary.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    input_shape : Tuple[int, ...]
        Input shape (C, H, W) without batch dimension
    device : str
        Device to run on
        
    Returns
    -------
    str
        Formatted summary string
        
    Examples
    --------
    >>> model = SimpleCNN(n_classes=10)
    >>> summary = model_summary(model, (3, 32, 32))
    >>> print(summary)
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(1, *input_shape).to(device)
    
    # Forward pass to get layer outputs
    summary_str = []
    summary_str.append("=" * 70)
    summary_str.append(f"{'Layer':<30} {'Output Shape':<20} {'Params':<15}")
    summary_str.append("=" * 70)
    
    total_params = 0
    
    def hook_fn(module, input, output):
        nonlocal total_params
        
        class_name = module.__class__.__name__
        output_shape = str(list(output.shape))
        
        params = sum(p.numel() for p in module.parameters())
        total_params += params
        
        summary_str.append(f"{class_name:<30} {output_shape:<20} {params:<15,}")
    
    # Register hooks
    hooks = []
    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, type(model)):
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    summary_str.append("=" * 70)
    summary_str.append(f"Total parameters: {total_params:,}")
    summary_str.append("=" * 70)
    
    return "\n".join(summary_str)


if __name__ == "__main__":
    # Quick test
    print("Testing architectures.py...")
    
    # Test SimpleCNN
    model = SimpleCNN(n_classes=10, input_channels=3)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"SimpleCNN output shape: {y.shape}")
    
    # Test LeNet5
    model_lenet = LeNet5(n_classes=10, input_channels=1)
    x_lenet = torch.randn(2, 1, 28, 28)
    y_lenet = model_lenet(x_lenet)
    print(f"LeNet5 output shape: {y_lenet.shape}")
    
    # Test parameter counting
    params = count_parameters(model_lenet)
    print(f"LeNet5 parameters: {params['total']:,}")
    
    # Test ResidualBlock
    res_block = ResidualBlock(64, 64)
    x_res = torch.randn(2, 64, 32, 32)
    y_res = res_block(x_res)
    print(f"ResidualBlock output shape: {y_res.shape}")
    
    print("✓ All tests passed!")
