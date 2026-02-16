"""
Civil Engineering example datasets for CNN demonstrations.

Generates synthetic CE datasets for crack detection, land use classification,
pavement distress, and structural health monitoring.
"""

import numpy as np
from typing import Tuple, List
from scipy import signal


def generate_crack_dataset(n_per_class: int = 100, img_size: int = 64,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic crack detection dataset.
    
    Creates procedural concrete textures with/without crack patterns.
    
    Parameters
    ----------
    n_per_class : int
        Number of images per class (cracked/uncracked)
    img_size : int
        Image size (square)
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (images, labels) where images shape is (n_total, img_size, img_size, 3)
        and labels are 0=uncracked, 1=cracked
    """
    np.random.seed(random_state)
    
    n_total = n_per_class * 2
    images = np.zeros((n_total, img_size, img_size, 3))
    labels = np.zeros(n_total, dtype=np.int64)
    
    for i in range(n_total):
        # Base concrete texture (grainy)
        base = np.random.rand(img_size, img_size) * 0.3 + 0.5
        
        # Add noise for concrete texture
        noise = np.random.randn(img_size, img_size) * 0.1
        texture = base + noise
        
        # Cracked class
        if i >= n_per_class:
            labels[i] = 1
            
            # Add crack pattern (random line)
            crack_start = (np.random.randint(0, img_size//2),
                          np.random.randint(0, img_size))
            crack_end = (np.random.randint(img_size//2, img_size),
                        np.random.randint(0, img_size))
            
            # Draw crack as dark line
            rr, cc = _draw_line(crack_start, crack_end, img_size)
            texture[rr, cc] *= 0.3
            
            # Add some branching
            if np.random.rand() > 0.5:
                branch_start = (rr[len(rr)//2], cc[len(cc)//2])
                branch_end = (branch_start[0] + np.random.randint(-10, 10),
                             branch_start[1] + np.random.randint(-10, 10))
                rr2, cc2 = _draw_line(branch_start, branch_end, img_size)
                texture[rr2, cc2] *= 0.4
        
        # Convert to RGB (grayscale replicated)
        texture = np.clip(texture, 0, 1)
        images[i] = np.stack([texture, texture, texture], axis=-1)
    
    return images.astype(np.float32), labels


def generate_land_use_dataset(n_per_class: int = 100, img_size: int = 64,
                              random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic land use classification dataset.
    
    Classes: 0=urban, 1=vegetation, 2=water, 3=bare_soil
    
    Parameters
    ----------
    n_per_class : int
        Number of images per class
    img_size : int
        Image size
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (images, labels)
    """
    np.random.seed(random_state)
    
    n_classes = 4
    n_total = n_per_class * n_classes
    images = np.zeros((n_total, img_size, img_size, 3))
    labels = np.zeros(n_total, dtype=np.int64)
    
    class_colors = {
        0: ([0.5, 0.5, 0.5], 0.2),  # Urban: gray
        1: ([0.2, 0.6, 0.2], 0.15),  # Vegetation: green
        2: ([0.2, 0.4, 0.8], 0.1),   # Water: blue
        3: ([0.7, 0.5, 0.3], 0.2)    # Bare soil: brown
    }
    
    for i in range(n_total):
        class_idx = i // n_per_class
        labels[i] = class_idx
        
        base_color, noise_level = class_colors[class_idx]
        
        # Create base color with texture
        for c in range(3):
            channel = np.ones((img_size, img_size)) * base_color[c]
            channel += np.random.randn(img_size, img_size) * noise_level
            images[i, :, :, c] = channel
        
        # Add class-specific patterns
        if class_idx == 0:  # Urban: add grid pattern
            images[i, ::8, :, :] *= 0.7
            images[i, :, ::8, :] *= 0.7
        elif class_idx == 1:  # Vegetation: add patches
            for _ in range(5):
                y, x = np.random.randint(0, img_size-10, 2)
                images[i, y:y+10, x:x+10, 1] *= 1.2
        elif class_idx == 2:  # Water: smooth
            from scipy.ndimage import gaussian_filter
            for c in range(3):
                images[i, :, :, c] = gaussian_filter(images[i, :, :, c], sigma=2)
    
    images = np.clip(images, 0, 1)
    return images.astype(np.float32), labels


def generate_pavement_distress_dataset(n_per_class: int = 100, img_size: int = 64,
                                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic pavement distress dataset.
    
    Classes: 0=good, 1=alligator_cracking, 2=pothole, 3=rutting
    
    Parameters
    ----------
    n_per_class : int
        Number of images per class
    img_size : int
        Image size
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (images, labels)
    """
    np.random.seed(random_state)
    
    n_classes = 4
    n_total = n_per_class * n_classes
    images = np.zeros((n_total, img_size, img_size, 3))
    labels = np.zeros(n_total, dtype=np.int64)
    
    for i in range(n_total):
        class_idx = i // n_per_class
        labels[i] = class_idx
        
        # Base asphalt texture
        base = np.random.rand(img_size, img_size) * 0.2 + 0.3
        
        if class_idx == 0:  # Good pavement
            pass  # Just base texture
        elif class_idx == 1:  # Alligator cracking (network of cracks)
            for _ in range(5):
                start = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                end = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                rr, cc = _draw_line(start, end, img_size)
                base[rr, cc] *= 0.5
        elif class_idx == 2:  # Pothole (dark circular region)
            center = (np.random.randint(img_size//4, 3*img_size//4),
                     np.random.randint(img_size//4, 3*img_size//4))
            radius = np.random.randint(5, 15)
            y, x = np.ogrid[:img_size, :img_size]
            mask = (y - center[0])**2 + (x - center[1])**2 <= radius**2
            base[mask] *= 0.3
        elif class_idx == 3:  # Rutting (parallel depressions)
            for offset in [img_size//3, 2*img_size//3]:
                base[:, offset-2:offset+2] *= 0.6
        
        # Convert to RGB
        base = np.clip(base, 0, 1)
        images[i] = np.stack([base, base, base], axis=-1)
    
    return images.astype(np.float32), labels


def generate_vibration_signals(n_signals: int = 100, length: int = 1000,
                               damage_level: float = 0.0,
                               random_state: int = 42) -> np.ndarray:
    """
    Generate synthetic structural vibration signals.
    
    Parameters
    ----------
    n_signals : int
        Number of signals to generate
    length : int
        Signal length (time steps)
    damage_level : float
        Damage level (0=healthy, 1=severely damaged)
    random_state : int
        Random seed
        
    Returns
    -------
    np.ndarray
        Vibration signals, shape (n_signals, length)
    """
    np.random.seed(random_state)
    
    signals = np.zeros((n_signals, length))
    t = np.linspace(0, 10, length)
    
    for i in range(n_signals):
        # Base frequency (natural frequency)
        f0 = 5.0 + np.random.randn() * 0.5
        
        # Damage shifts frequency and adds nonlinearity
        f_damaged = f0 * (1 - damage_level * 0.2)
        
        # Generate signal
        sig = np.sin(2 * np.pi * f_damaged * t)
        
        # Add harmonics
        sig += 0.3 * np.sin(2 * np.pi * 2 * f_damaged * t)
        sig += 0.1 * np.sin(2 * np.pi * 3 * f_damaged * t)
        
        # Damage adds nonlinearity and noise
        if damage_level > 0:
            sig += damage_level * 0.5 * np.sin(2 * np.pi * f_damaged * t)**3
            sig += damage_level * np.random.randn(length) * 0.2
        
        # Add measurement noise
        sig += np.random.randn(length) * 0.05
        
        # Exponential decay (damping)
        damping = np.exp(-t * (0.1 + damage_level * 0.1))
        sig *= damping
        
        signals[i] = sig
    
    return signals.astype(np.float32)


def make_spectrogram(signal: np.ndarray, fs: int = 100,
                    nperseg: int = 64) -> np.ndarray:
    """
    Convert 1D signal to 2D spectrogram using STFT.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal, shape (length,)
    fs : int
        Sampling frequency
    nperseg : int
        Length of each segment for STFT
        
    Returns
    -------
    np.ndarray
        Spectrogram, shape (freq_bins, time_bins)
    """
    from scipy.signal import spectrogram
    
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    return Sxx_db.astype(np.float32)


def _draw_line(start: Tuple[int, int], end: Tuple[int, int],
              img_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to draw a line (Bresenham's algorithm simplified).
    
    Returns row and column indices of line pixels.
    """
    y0, x0 = start
    y1, x1 = end
    
    # Clip to image bounds
    y0, x0 = np.clip([y0, x0], 0, img_size-1)
    y1, x1 = np.clip([y1, x1], 0, img_size-1)
    
    # Simple line interpolation
    n_points = max(abs(y1 - y0), abs(x1 - x0)) + 1
    yy = np.linspace(y0, y1, n_points).astype(int)
    xx = np.linspace(x0, x1, n_points).astype(int)
    
    # Clip again to be safe
    yy = np.clip(yy, 0, img_size-1)
    xx = np.clip(xx, 0, img_size-1)
    
    return yy, xx


if __name__ == "__main__":
    print("Testing ce_examples.py...")
    
    # Test crack dataset
    images, labels = generate_crack_dataset(n_per_class=10, img_size=32)
    print(f"Crack dataset: {images.shape}, {labels.shape}")
    print(f"  Class distribution: {np.bincount(labels)}")
    
    # Test land use dataset
    images, labels = generate_land_use_dataset(n_per_class=10, img_size=32)
    print(f"Land use dataset: {images.shape}, {labels.shape}")
    print(f"  Class distribution: {np.bincount(labels)}")
    
    # Test vibration signals
    signals = generate_vibration_signals(n_signals=10, length=100, damage_level=0.5)
    print(f"Vibration signals: {signals.shape}")
    
    # Test spectrogram
    spec = make_spectrogram(signals[0])
    print(f"Spectrogram: {spec.shape}")
    
    print("✓ All CE examples tests passed!")
