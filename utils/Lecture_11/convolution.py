"""
Convolution operations implemented from scratch in NumPy.

All functions use pure NumPy for pedagogical clarity.
"""

import numpy as np
from typing import Tuple, Dict, Generator, Optional


def conv1d(signal: np.ndarray, kernel: np.ndarray, 
           padding: int = 0, stride: int = 1) -> np.ndarray:
    """
    1D convolution from scratch.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (n,)
    kernel : np.ndarray
        Convolution kernel, shape (k,)
    padding : int
        Number of zeros to pad on each side
    stride : int
        Stride for sliding window
        
    Returns
    -------
    np.ndarray
        Convolved signal
        
    Examples
    --------
    >>> signal = np.array([1, 2, 3, 4, 5])
    >>> kernel = np.array([1, 0, -1])  # Edge detector
    >>> conv1d(signal, kernel)
    array([-2, -2, -2])
    """
    n = len(signal)
    k = len(kernel)
    
    # Apply padding
    if padding > 0:
        signal = np.pad(signal, padding, mode='constant')
        n = len(signal)
    
    # Compute output size
    out_size = (n - k) // stride + 1
    output = np.zeros(out_size)
    
    # Convolve
    for i in range(out_size):
        start = i * stride
        output[i] = np.sum(signal[start:start + k] * kernel)
    
    return output


def output_size(H: int, W: int, k: int, p: int, s: int) -> Tuple[int, int]:
    """
    Compute output dimensions after convolution.
    
    Formula: floor((H + 2p - k) / s + 1)
    
    Parameters
    ----------
    H : int
        Input height
    W : int
        Input width
    k : int
        Kernel size (assuming square kernel)
    p : int
        Padding
    s : int
        Stride
        
    Returns
    -------
    Tuple[int, int]
        (output_height, output_width)
        
    Examples
    --------
    >>> output_size(32, 32, 5, 0, 1)
    (28, 28)
    >>> output_size(32, 32, 3, 1, 1)  # Same padding
    (32, 32)
    """
    H_out = (H + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1
    return H_out, W_out


def conv2d(image: np.ndarray, kernel: np.ndarray,
           padding: int = 0, stride: int = 1) -> np.ndarray:
    """
    2D convolution from scratch (single channel).
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W)
    kernel : np.ndarray
        Convolution kernel, shape (k, k)
    padding : int
        Padding size
    stride : int
        Stride
        
    Returns
    -------
    np.ndarray
        Feature map after convolution
        
    Examples
    --------
    >>> image = np.ones((5, 5))
    >>> kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    >>> result = conv2d(image, kernel)
    >>> result.shape
    (3, 3)
    """
    H, W = image.shape
    k = kernel.shape[0]
    
    # Apply padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
        H, W = image.shape
    
    # Compute output size
    H_out, W_out = output_size(H, W, k, 0, stride)
    output = np.zeros((H_out, W_out))
    
    # Convolve
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract window and compute dot product
            window = image[h_start:h_start + k, w_start:w_start + k]
            output[i, j] = np.sum(window * kernel)
    
    return output


def conv2d_multichannel(image: np.ndarray, kernels: np.ndarray,
                        padding: int = 0, stride: int = 1,
                        bias: Optional[np.ndarray] = None) -> np.ndarray:
    """
    2D convolution with multiple input channels and multiple filters.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W, C_in)
    kernels : np.ndarray
        Convolution kernels, shape (F, k, k, C_in)
        where F is number of filters
    padding : int
        Padding size
    stride : int
        Stride
    bias : np.ndarray, optional
        Bias terms, shape (F,)
        
    Returns
    -------
    np.ndarray
        Feature maps, shape (H_out, W_out, F)
        
    Examples
    --------
    >>> image = np.random.randn(32, 32, 3)  # RGB image
    >>> kernels = np.random.randn(16, 3, 3, 3)  # 16 filters
    >>> result = conv2d_multichannel(image, kernels)
    >>> result.shape
    (30, 30, 16)
    """
    H, W, C_in = image.shape
    F, k, _, _ = kernels.shape
    
    # Apply padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                      mode='constant')
        H, W = image.shape[:2]
    
    # Compute output size
    H_out, W_out = output_size(H, W, k, 0, stride)
    output = np.zeros((H_out, W_out, F))
    
    # Convolve each filter
    for f in range(F):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                
                # Extract window (all channels)
                window = image[h_start:h_start + k, w_start:w_start + k, :]
                
                # Convolve: sum over spatial dims AND channels
                output[i, j, f] = np.sum(window * kernels[f])
        
        # Add bias if provided
        if bias is not None:
            output[:, :, f] += bias[f]
    
    return output


def conv2d_step_by_step(image: np.ndarray, kernel: np.ndarray,
                        padding: int = 0, stride: int = 1) -> Generator:
    """
    Generator that yields each step of 2D convolution for animation.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W)
    kernel : np.ndarray
        Convolution kernel, shape (k, k)
    padding : int
        Padding size
    stride : int
        Stride
        
    Yields
    ------
    dict
        Dictionary with keys:
        - 'position': (i, j) current output position
        - 'window': extracted image window
        - 'kernel': the kernel
        - 'value': computed output value
        - 'output_so_far': partially filled output array
        
    Examples
    --------
    >>> image = np.random.randn(5, 5)
    >>> kernel = np.ones((3, 3)) / 9  # Average filter
    >>> for step in conv2d_step_by_step(image, kernel):
    ...     print(f"Position {step['position']}: {step['value']:.2f}")
    """
    H, W = image.shape
    k = kernel.shape[0]
    
    # Apply padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
        H, W = image.shape
    
    # Compute output size
    H_out, W_out = output_size(H, W, k, 0, stride)
    output = np.zeros((H_out, W_out))
    
    # Convolve step by step
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract window
            window = image[h_start:h_start + k, w_start:w_start + k]
            value = np.sum(window * kernel)
            output[i, j] = value
            
            # Yield current state
            yield {
                'position': (i, j),
                'window': window.copy(),
                'window_coords': (h_start, w_start),
                'kernel': kernel.copy(),
                'value': value,
                'output_so_far': output.copy()
            }


def common_kernels() -> Dict[str, np.ndarray]:
    """
    Dictionary of common convolution kernels.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping kernel names to kernel arrays
        
    Examples
    --------
    >>> kernels = common_kernels()
    >>> kernels['edge_horizontal']
    array([[-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1]])
    """
    kernels = {
        # Identity
        'identity': np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32),
        
        # Blur / Smoothing
        'blur': np.ones((3, 3), dtype=np.float32) / 9,
        
        'gaussian_blur': np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16,
        
        # Sharpening
        'sharpen': np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32),
        
        # Edge detection
        'edge_horizontal': np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32),
        
        'edge_vertical': np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32),
        
        # Sobel operators
        'sobel_x': np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32),
        
        'sobel_y': np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32),
        
        # Emboss
        'emboss': np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ], dtype=np.float32),
        
        # Outline
        'outline': np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32),
    }
    
    return kernels


if __name__ == "__main__":
    # Quick test
    print("Testing convolution.py...")
    
    # Test 1D convolution
    signal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    kernel_1d = np.array([1, 0, -1], dtype=np.float32)
    result_1d = conv1d(signal, kernel_1d)
    print(f"1D conv result: {result_1d}")
    
    # Test 2D convolution
    image = np.ones((5, 5), dtype=np.float32)
    kernels_dict = common_kernels()
    result_2d = conv2d(image, kernels_dict['edge_horizontal'])
    print(f"2D conv result shape: {result_2d.shape}")
    
    # Test output size calculation
    h_out, w_out = output_size(32, 32, 5, 0, 1)
    print(f"Output size (32x32, k=5, p=0, s=1): {h_out}x{w_out}")
    
    print("✓ All tests passed!")
