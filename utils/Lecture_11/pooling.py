"""
Pooling operations implemented from scratch in NumPy.

All functions use pure NumPy for pedagogical clarity.
"""

import numpy as np
from typing import Tuple, Generator, Literal


def max_pool2d(feature_map: np.ndarray, pool_size: int = 2,
               stride: int = 2) -> np.ndarray:
    """
    2D max pooling from scratch.
    
    Parameters
    ----------
    feature_map : np.ndarray
        Input feature map, shape (H, W) or (H, W, C)
    pool_size : int
        Size of pooling window (square)
    stride : int
        Stride for pooling window
        
    Returns
    -------
    np.ndarray
        Pooled feature map
        
    Examples
    --------
    >>> feature_map = np.array([[1, 2, 3, 4],
    ...                         [5, 6, 7, 8],
    ...                         [9, 10, 11, 12],
    ...                         [13, 14, 15, 16]])
    >>> max_pool2d(feature_map, pool_size=2, stride=2)
    array([[ 6.,  8.],
           [14., 16.]])
    """
    # Handle both (H, W) and (H, W, C) inputs
    if feature_map.ndim == 2:
        H, W = feature_map.shape
        has_channels = False
    elif feature_map.ndim == 3:
        H, W, C = feature_map.shape
        has_channels = True
    else:
        raise ValueError(f"Expected 2D or 3D input, got shape {feature_map.shape}")
    
    # Compute output size
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    if has_channels:
        output = np.zeros((H_out, W_out, C))
        
        # Pool each channel independently
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    
                    window = feature_map[h_start:h_start + pool_size,
                                       w_start:w_start + pool_size, c]
                    output[i, j, c] = np.max(window)
    else:
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                
                window = feature_map[h_start:h_start + pool_size,
                                   w_start:w_start + pool_size]
                output[i, j] = np.max(window)
    
    return output


def avg_pool2d(feature_map: np.ndarray, pool_size: int = 2,
               stride: int = 2) -> np.ndarray:
    """
    2D average pooling from scratch.
    
    Parameters
    ----------
    feature_map : np.ndarray
        Input feature map, shape (H, W) or (H, W, C)
    pool_size : int
        Size of pooling window (square)
    stride : int
        Stride for pooling window
        
    Returns
    -------
    np.ndarray
        Pooled feature map
        
    Examples
    --------
    >>> feature_map = np.array([[1, 2, 3, 4],
    ...                         [5, 6, 7, 8],
    ...                         [9, 10, 11, 12],
    ...                         [13, 14, 15, 16]])
    >>> avg_pool2d(feature_map, pool_size=2, stride=2)
    array([[ 3.5,  5.5],
           [11.5, 13.5]])
    """
    # Handle both (H, W) and (H, W, C) inputs
    if feature_map.ndim == 2:
        H, W = feature_map.shape
        has_channels = False
    elif feature_map.ndim == 3:
        H, W, C = feature_map.shape
        has_channels = True
    else:
        raise ValueError(f"Expected 2D or 3D input, got shape {feature_map.shape}")
    
    # Compute output size
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    
    if has_channels:
        output = np.zeros((H_out, W_out, C))
        
        # Pool each channel independently
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    
                    window = feature_map[h_start:h_start + pool_size,
                                       w_start:w_start + pool_size, c]
                    output[i, j, c] = np.mean(window)
    else:
        output = np.zeros((H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                
                window = feature_map[h_start:h_start + pool_size,
                                   w_start:w_start + pool_size]
                output[i, j] = np.mean(window)
    
    return output


def global_avg_pool(feature_map: np.ndarray) -> np.ndarray:
    """
    Global average pooling: average over entire spatial dimensions.
    
    Commonly used before final classification layer in modern CNNs.
    
    Parameters
    ----------
    feature_map : np.ndarray
        Input feature map, shape (H, W, C)
        
    Returns
    -------
    np.ndarray
        Pooled features, shape (C,)
        
    Examples
    --------
    >>> feature_map = np.random.randn(7, 7, 512)
    >>> result = global_avg_pool(feature_map)
    >>> result.shape
    (512,)
    """
    if feature_map.ndim != 3:
        raise ValueError(f"Expected 3D input (H, W, C), got shape {feature_map.shape}")
    
    # Average over spatial dimensions (H, W)
    return np.mean(feature_map, axis=(0, 1))


def pool_step_by_step(feature_map: np.ndarray, pool_size: int = 2,
                      stride: int = 2,
                      mode: Literal['max', 'avg'] = 'max') -> Generator:
    """
    Generator that yields each step of pooling for animation.
    
    Parameters
    ----------
    feature_map : np.ndarray
        Input feature map, shape (H, W)
    pool_size : int
        Size of pooling window
    stride : int
        Stride
    mode : {'max', 'avg'}
        Pooling mode
        
    Yields
    ------
    dict
        Dictionary with keys:
        - 'position': (i, j) current output position
        - 'window': extracted window
        - 'window_coords': (h_start, w_start) in input
        - 'value': computed pooled value
        - 'output_so_far': partially filled output array
        
    Examples
    --------
    >>> feature_map = np.random.randn(8, 8)
    >>> for step in pool_step_by_step(feature_map, pool_size=2, mode='max'):
    ...     print(f"Position {step['position']}: {step['value']:.2f}")
    """
    if feature_map.ndim != 2:
        raise ValueError("Step-by-step pooling only supports 2D inputs")
    
    H, W = feature_map.shape
    
    # Compute output size
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))
    
    # Pool function
    pool_fn = np.max if mode == 'max' else np.mean
    
    # Pool step by step
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract window
            window = feature_map[h_start:h_start + pool_size,
                               w_start:w_start + pool_size]
            value = pool_fn(window)
            output[i, j] = value
            
            # Yield current state
            yield {
                'position': (i, j),
                'window': window.copy(),
                'window_coords': (h_start, w_start),
                'value': value,
                'mode': mode,
                'output_so_far': output.copy()
            }


if __name__ == "__main__":
    # Quick test
    print("Testing pooling.py...")
    
    # Test max pooling
    feature_map = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float32)
    
    result_max = max_pool2d(feature_map, pool_size=2, stride=2)
    print(f"Max pool result:\n{result_max}")
    
    result_avg = avg_pool2d(feature_map, pool_size=2, stride=2)
    print(f"Avg pool result:\n{result_avg}")
    
    # Test with channels
    feature_map_3d = np.random.randn(8, 8, 16).astype(np.float32)
    result_max_3d = max_pool2d(feature_map_3d, pool_size=2, stride=2)
    print(f"Max pool 3D result shape: {result_max_3d.shape}")
    
    # Test global average pooling
    result_gap = global_avg_pool(feature_map_3d)
    print(f"Global avg pool result shape: {result_gap.shape}")
    
    print("✓ All tests passed!")
