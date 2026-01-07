"""
Civil Engineering application examples for Linear Algebra concepts.

This module provides CE-specific examples including structural stiffness matrices,
vibration systems, force decomposition, and sensor data generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def create_stiffness_matrix(n_dof=4, k_values=None):
    """
    Create a structural stiffness matrix for a simple truss or beam system.
    
    Parameters:
    -----------
    n_dof : int
        Number of degrees of freedom
    k_values : array-like, optional
        Stiffness values for each element
    
    Returns:
    --------
    K : ndarray
        Stiffness matrix
    info : dict
        Information about the system
    """
    if k_values is None:
        k_values = np.random.uniform(1e6, 5e6, n_dof)
    
    # Create a simple tridiagonal stiffness matrix (spring system)
    K = np.zeros((n_dof, n_dof))
    
    for i in range(n_dof):
        K[i, i] = k_values[i] + (k_values[i-1] if i > 0 else 0)
        if i > 0:
            K[i, i-1] = -k_values[i-1]
            K[i-1, i] = -k_values[i-1]
    
    info = {
        'n_dof': n_dof,
        'k_values': k_values,
        'condition_number': np.linalg.cond(K),
        'rank': np.linalg.matrix_rank(K),
        'symmetric': np.allclose(K, K.T),
        'positive_definite': np.all(np.linalg.eigvals(K) > 0)
    }
    
    return K, info


def create_vibration_system(n_masses=3, mass_values=None, k_values=None):
    """
    Create a mass-spring vibration system for eigenvalue analysis.
    
    Parameters:
    -----------
    n_masses : int
        Number of masses in the system
    mass_values : array-like, optional
        Mass values (kg)
    k_values : array-like, optional
        Spring stiffness values (N/m)
    
    Returns:
    --------
    M : ndarray
        Mass matrix
    K : ndarray
        Stiffness matrix
    eigenvalues : ndarray
        Natural frequencies squared (ω²)
    eigenvectors : ndarray
        Mode shapes
    """
    if mass_values is None:
        mass_values = np.ones(n_masses) * 1000  # 1000 kg each
    if k_values is None:
        k_values = np.ones(n_masses + 1) * 1e6  # 1 MN/m
    
    # Mass matrix (diagonal)
    M = np.diag(mass_values)
    
    # Stiffness matrix (tridiagonal)
    K = np.zeros((n_masses, n_masses))
    for i in range(n_masses):
        K[i, i] = k_values[i] + k_values[i+1]
        if i > 0:
            K[i, i-1] = -k_values[i]
            K[i-1, i] = -k_values[i]
    
    # Solve eigenvalue problem: K*phi = ω²*M*phi
    # Transform to standard form: M^(-1)*K*phi = ω²*phi
    M_inv = np.linalg.inv(M)
    A = M_inv @ K
    
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by frequency
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Natural frequencies in Hz
    natural_frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    
    return M, K, eigenvalues, eigenvectors, natural_frequencies


def simulate_structural_modes(eigenvectors, n_frames=100, mode_index=0):
    """
    Animate structural vibration mode shapes.
    
    Parameters:
    -----------
    eigenvectors : ndarray
        Mode shape vectors (columns are modes)
    n_frames : int
        Number of animation frames
    mode_index : int
        Which mode to animate
    
    Returns:
    --------
    HTML animation object
    """
    n_masses = eigenvectors.shape[0]
    mode_shape = eigenvectors[:, mode_index]
    
    # Normalize mode shape
    mode_shape = mode_shape / np.max(np.abs(mode_shape))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        t = frame / n_frames * 2 * np.pi
        displacement = mode_shape * np.sin(t)
        
        # Plot 1: Physical representation
        positions = np.arange(n_masses)
        ax1.plot(positions, np.zeros(n_masses), 'k--', alpha=0.3, label='Equilibrium')
        ax1.plot(positions, displacement, 'bo-', markersize=15, linewidth=2, label='Current position')
        
        for i, (pos, disp) in enumerate(zip(positions, displacement)):
            ax1.plot([pos, pos], [0, disp], 'r--', alpha=0.5, linewidth=1)
            ax1.text(pos, -0.3, f'Mass {i+1}', ha='center', fontsize=10)
        
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_xlabel('Mass Position', fontsize=12)
        ax1.set_ylabel('Displacement', fontsize=12)
        ax1.set_title(f'Mode {mode_index + 1} - Physical Displacement', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mode shape
        ax2.bar(range(n_masses), mode_shape, color='blue', alpha=0.6, label='Mode shape')
        ax2.bar(range(n_masses), displacement, color='red', alpha=0.6, label='Current displacement')
        ax2.set_xlabel('Mass Index', fontsize=12)
        ax2.set_ylabel('Amplitude', fontsize=12)
        ax2.set_title(f'Mode Shape Components (t = {t:.2f} rad)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=50, blit=True, repeat=True)
    plt.close()
    
    return HTML(anim.to_jshtml())


def generate_sensor_data(n_sensors=10, n_timesteps=100, n_modes=3, noise_level=0.1):
    """
    Generate synthetic structural health monitoring sensor data.
    
    Parameters:
    -----------
    n_sensors : int
        Number of sensors
    n_timesteps : int
        Number of time steps
    n_modes : int
        Number of underlying vibration modes
    noise_level : float
        Noise standard deviation
    
    Returns:
    --------
    data : ndarray
        Sensor data matrix (n_timesteps × n_sensors)
    true_modes : ndarray
        True mode shapes
    """
    # Generate random mode shapes
    true_modes = np.random.randn(n_sensors, n_modes)
    
    # Orthogonalize modes (Gram-Schmidt)
    for i in range(n_modes):
        for j in range(i):
            true_modes[:, i] -= np.dot(true_modes[:, i], true_modes[:, j]) * true_modes[:, j]
        true_modes[:, i] /= np.linalg.norm(true_modes[:, i])
    
    # Generate time-varying amplitudes
    t = np.linspace(0, 10, n_timesteps)
    frequencies = np.array([0.5, 1.2, 2.3])[:n_modes]
    amplitudes = np.array([np.sin(2 * np.pi * f * t) for f in frequencies]).T
    
    # Combine modes
    data = amplitudes @ true_modes.T
    
    # Add noise
    data += noise_level * np.random.randn(n_timesteps, n_sensors)
    
    return data, true_modes


def create_force_decomposition(force_magnitude=1000, force_angle_deg=30, 
                               beam_angle_deg=0, figsize=(10, 8)):
    """
    Decompose a force into components along beam axes.
    
    Parameters:
    -----------
    force_magnitude : float
        Magnitude of applied force (N)
    force_angle_deg : float
        Angle of force from horizontal (degrees)
    beam_angle_deg : float
        Angle of beam from horizontal (degrees)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    components : dict
        Force components
    """
    # Convert to radians
    force_angle = np.deg2rad(force_angle_deg)
    beam_angle = np.deg2rad(beam_angle_deg)
    
    # Force vector
    force = force_magnitude * np.array([np.cos(force_angle), np.sin(force_angle)])
    
    # Beam direction (axial)
    beam_dir = np.array([np.cos(beam_angle), np.sin(beam_angle)])
    
    # Perpendicular direction (transverse)
    perp_dir = np.array([-np.sin(beam_angle), np.cos(beam_angle)])
    
    # Decompose force
    axial_component = np.dot(force, beam_dir) * beam_dir
    transverse_component = np.dot(force, perp_dir) * perp_dir
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw beam
    beam_length = 3
    beam_end = beam_length * beam_dir
    ax.plot([0, beam_end[0]], [0, beam_end[1]], 'k-', linewidth=8, label='Beam', alpha=0.6)
    
    # Draw force vectors
    scale = 0.003  # Scale for visualization
    
    # Applied force
    ax.arrow(0, 0, force[0]*scale, force[1]*scale, head_width=0.15, head_length=0.2,
            fc='red', ec='red', linewidth=3, label=f'Applied Force ({force_magnitude} N)', zorder=5)
    
    # Axial component
    ax.arrow(0, 0, axial_component[0]*scale, axial_component[1]*scale,
            head_width=0.12, head_length=0.15, fc='blue', ec='blue', linewidth=2.5,
            linestyle='--', label=f'Axial: {np.linalg.norm(axial_component):.1f} N', zorder=4)
    
    # Transverse component
    ax.arrow(axial_component[0]*scale, axial_component[1]*scale,
            transverse_component[0]*scale, transverse_component[1]*scale,
            head_width=0.12, head_length=0.15, fc='green', ec='green', linewidth=2.5,
            linestyle='--', label=f'Transverse: {np.linalg.norm(transverse_component):.1f} N', zorder=4)
    
    # Draw right angle indicator
    if np.linalg.norm(transverse_component) > 10:
        indicator_size = 0.2
        corner = axial_component * scale + indicator_size * (beam_dir + perp_dir)
        square = np.array([axial_component * scale,
                          axial_component * scale + indicator_size * beam_dir,
                          corner,
                          axial_component * scale + indicator_size * perp_dir,
                          axial_component * scale])
        ax.plot(square[:, 0], square[:, 1], 'k-', linewidth=1, alpha=0.5)
    
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 3)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('Force Decomposition on Beam', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add angle annotations
    angle_arc = plt.Circle((0, 0), 0.5, fill=False, color='red', linestyle=':', linewidth=1.5)
    ax.add_patch(angle_arc)
    ax.text(0.6, 0.3, f'{force_angle_deg}°', fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    components = {
        'force': force,
        'axial': axial_component,
        'transverse': transverse_component,
        'axial_magnitude': np.linalg.norm(axial_component),
        'transverse_magnitude': np.linalg.norm(transverse_component)
    }
    
    return fig, ax, components
