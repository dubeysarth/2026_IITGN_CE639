"""
Visualization functions for vector calculus and backpropagation.

This module provides plotting utilities for gradients, Jacobians,
computation graphs, and neural network training.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


def plot_gradient_field(f, x_range, y_range, resolution=20, title="Gradient Field"):
    """
    Plot gradient vector field for a 2D function.
    
    Parameters:
    -----------
    f : callable
        Scalar function f(x, y)
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Grid resolution
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    from .calculus import gradient
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    Z = np.zeros_like(X)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = f(point)
            grad = gradient(f, point)
            U[i, j] = grad[0]
            V[i, j] = grad[1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot
    contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
    ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.3)
    
    # Gradient field
    ax.quiver(X, Y, U, V, color='red', alpha=0.7, scale=50, width=0.003)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(contour, ax=ax, label='f(x, y)')
    
    return fig, ax


def plot_computation_graph(graph, figsize=(12, 8), title="Computation Graph"):
    """
    Visualize computation graph using networkx.
    
    Parameters:
    -----------
    graph : ComputationGraph
        Graph to visualize
    figsize : tuple
        Figure size
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    graph_data = graph.visualize()
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_name, info in graph_data['nodes'].items():
        G.add_node(node_name, **info)
    
    # Add edges
    for parent, child in graph_data['edges']:
        G.add_edge(parent, child)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if 'x' in node or 'const' in node:
            node_colors.append('lightblue')
        elif node == graph.output.name:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgreen')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=2000, alpha=0.9, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        info = graph_data['nodes'][node]
        labels[node] = f"{node}\nv={info['value']:.2f}\ng={info['grad']:.2f}"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig, ax


def animate_backprop(graph, interval=500):
    """
    Animate backpropagation through computation graph.
    
    Parameters:
    -----------
    graph : ComputationGraph
        Graph to animate
    interval : int
        Animation interval (ms)
    
    Returns:
    --------
    HTML animation
    """
    # This is a simplified version - full implementation would
    # show gradients flowing backward step by step
    
    fig, ax = plot_computation_graph(graph, title="Backpropagation Animation")
    plt.close()
    
    return HTML("<p>Animation: Gradients flow backward through graph</p>")


def plot_gradient_descent_3d(f, x_history, x_range, y_range, resolution=50):
    """
    Plot 3D surface with gradient descent path.
    
    Parameters:
    -----------
    f : callable
        Function f([x, y])
    x_history : array
        History of points (n_iters, 2)
    x_range : tuple
        (x_min, x_max)
    y_range : tuple
        (y_min, y_max)
    resolution : int
        Grid resolution
    
    Returns:
    --------
    fig, ax : matplotlib 3D objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, 
                          edgecolor='none')
    
    # Gradient descent path
    z_history = np.array([f(x) for x in x_history])
    ax.plot(x_history[:, 0], x_history[:, 1], z_history, 
           'ro-', linewidth=2, markersize=6, label='GD path')
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('f(x, y)', fontsize=11)
    ax.set_title('Gradient Descent on 3D Surface', fontsize=13, fontweight='bold')
    ax.legend()
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax


def plot_loss_during_training(loss_history, figsize=(10, 6)):
    """
    Plot training loss curve.
    
    Parameters:
    -----------
    loss_history : list
        Loss values over epochs
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(len(loss_history))
    ax.plot(epochs, loss_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return fig, ax


def plot_jacobian_heatmap(J, x_labels=None, y_labels=None, title="Jacobian Matrix"):
    """
    Plot Jacobian matrix as heatmap.
    
    Parameters:
    -----------
    J : array
        Jacobian matrix (m × n)
    x_labels : list, optional
        Labels for columns (inputs)
    y_labels : list, optional
        Labels for rows (outputs)
    title : str
        Plot title
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    m, n = J.shape
    
    if x_labels is None:
        x_labels = [f'x_{i+1}' for i in range(n)]
    if y_labels is None:
        y_labels = [f'f_{i+1}' for i in range(m)]
    
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, m)))
    
    im = ax.imshow(J, cmap='RdBu_r', aspect='auto', vmin=-np.abs(J).max(), 
                   vmax=np.abs(J).max())
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(m))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('∂f_i/∂x_j', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(m):
        for j in range(n):
            text = ax.text(j, i, f'{J[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_neural_network_architecture(layer_sizes, figsize=(12, 6)):
    """
    Visualize neural network architecture.
    
    Parameters:
    -----------
    layer_sizes : list
        List of layer sizes [n_in, n_h1, ..., n_out]
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    # Spacing
    layer_spacing = 1.0 / (n_layers - 1) if n_layers > 1 else 0.5
    
    # Draw neurons
    for i, n_neurons in enumerate(layer_sizes):
        x = i * layer_spacing
        neuron_spacing = 1.0 / (n_neurons + 1)
        
        for j in range(n_neurons):
            y = (j + 1) * neuron_spacing
            
            # Draw neuron
            circle = plt.Circle((x, y), 0.02, color='lightblue', 
                              ec='black', zorder=4)
            ax.add_patch(circle)
            
            # Draw connections to next layer
            if i < n_layers - 1:
                next_n_neurons = layer_sizes[i + 1]
                next_neuron_spacing = 1.0 / (next_n_neurons + 1)
                
                for k in range(next_n_neurons):
                    next_y = (k + 1) * next_neuron_spacing
                    ax.plot([x, x + layer_spacing], [y, next_y], 
                           'gray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Labels
    layer_names = ['Input'] + [f'Hidden {i}' for i in range(1, n_layers-1)] + ['Output']
    for i, name in enumerate(layer_names):
        x = i * layer_spacing
        ax.text(x, -0.1, name, ha='center', fontsize=11, fontweight='bold')
        ax.text(x, 1.1, f'{layer_sizes[i]} neurons', ha='center', fontsize=9)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_activation_functions(figsize=(14, 4)):
    """
    Plot common activation functions.
    
    Parameters:
    -----------
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib objects
    """
    from .neural_network import sigmoid, relu, tanh
    
    x = np.linspace(-5, 5, 200)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Sigmoid
    axes[0].plot(x, sigmoid(x), 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].set_xlabel('z', fontsize=11)
    axes[0].set_ylabel('σ(z)', fontsize=11)
    axes[0].set_title('Sigmoid', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # ReLU
    axes[1].plot(x, relu(x), 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel('z', fontsize=11)
    axes[1].set_ylabel('ReLU(z)', fontsize=11)
    axes[1].set_title('ReLU', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Tanh
    axes[2].plot(x, tanh(x), 'g-', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].axhline(1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].axhline(-1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[2].set_xlabel('z', fontsize=11)
    axes[2].set_ylabel('tanh(z)', fontsize=11)
    axes[2].set_title('Tanh', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
