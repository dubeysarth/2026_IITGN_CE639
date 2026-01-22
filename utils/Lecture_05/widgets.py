"""
Interactive widgets for exploring vector calculus concepts.

This module provides ipywidgets-based interactive controls for
gradients, chain rule, and backpropagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown, fixed
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


def gradient_explorer_widget():
    """
    Interactive widget to explore gradients on 2D surfaces.
    """
    def plot_gradient(x_val, y_val, function_type):
        from .calculus import gradient
        from .visualizations import plot_gradient_field
        
        # Define functions
        if function_type == 'Quadratic Bowl':
            f = lambda p: p[0]**2 + p[1]**2
            x_range, y_range = (-3, 3), (-3, 3)
        elif function_type == 'Saddle':
            f = lambda p: p[0]**2 - p[1]**2
            x_range, y_range = (-3, 3), (-3, 3)
        elif function_type == 'Rosenbrock':
            f = lambda p: (1 - p[0])**2 + 100*(p[1] - p[0]**2)**2
            x_range, y_range = (-2, 2), (-1, 3)
        
        # Plot gradient field
        fig, ax = plot_gradient_field(f, x_range, y_range, resolution=15,
                                      title=f'{function_type} Function')
        
        # Compute and plot gradient at selected point
        point = np.array([x_val, y_val])
        grad = gradient(f, point)
        f_val = f(point)
        
        # Plot point and gradient arrow
        ax.plot(x_val, y_val, 'ro', markersize=12, label='Selected point')
        ax.arrow(x_val, y_val, grad[0]*0.3, grad[1]*0.3, 
                head_width=0.15, head_length=0.1, fc='yellow', ec='yellow',
                linewidth=3, label='Gradient')
        ax.legend()
        
        plt.show()
        
        print(f"Point: ({x_val:.2f}, {y_val:.2f})")
        print(f"Function value: {f_val:.4f}")
        print(f"Gradient: [{grad[0]:.4f}, {grad[1]:.4f}]")
        print(f"Gradient magnitude: {np.linalg.norm(grad):.4f}")
    
    interact(plot_gradient,
            x_val=FloatSlider(min=-2, max=2, step=0.2, value=1.0,
                             description='x:',
                             style={'description_width': 'initial'}),
            y_val=FloatSlider(min=-2, max=2, step=0.2, value=1.0,
                             description='y:',
                             style={'description_width': 'initial'}),
            function_type=Dropdown(options=['Quadratic Bowl', 'Saddle', 'Rosenbrock'],
                                  value='Quadratic Bowl',
                                  description='Function:',
                                  style={'description_width': 'initial'}))


def chain_rule_widget():
    """
    Interactive widget to visualize chain rule.
    """
    def demonstrate_chain_rule(x_val):
        # Example: y = (x^2 + 1)^3
        # dy/dx = 3(x^2 + 1)^2 * 2x = 6x(x^2 + 1)^2
        
        # Inner function: u = x^2 + 1
        u = x_val**2 + 1
        du_dx = 2 * x_val
        
        # Outer function: y = u^3
        y = u**3
        dy_du = 3 * u**2
        
        # Chain rule: dy/dx = dy/du * du/dx
        dy_dx = dy_du * du_dx
        
        # Analytical derivative
        analytical = 6 * x_val * (x_val**2 + 1)**2
        
        # Plot
        x_range = np.linspace(-3, 3, 200)
        y_vals = (x_range**2 + 1)**3
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Function plot
        ax1.plot(x_range, y_vals, 'b-', linewidth=2, label='y = (x² + 1)³')
        ax1.plot(x_val, y, 'ro', markersize=10, label=f'Point ({x_val:.2f}, {y:.2f})')
        
        # Tangent line
        tangent_x = np.array([x_val - 0.5, x_val + 0.5])
        tangent_y = y + dy_dx * (tangent_x - x_val)
        ax1.plot(tangent_x, tangent_y, 'r--', linewidth=2, label='Tangent line')
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Function and Tangent', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chain rule visualization
        ax2.text(0.5, 0.9, 'Chain Rule Breakdown', ha='center', fontsize=14, 
                fontweight='bold', transform=ax2.transAxes)
        
        text = f"""
        Inner function: u = x² + 1
        u({x_val:.2f}) = {u:.4f}
        du/dx = 2x = {du_dx:.4f}
        
        Outer function: y = u³
        y = {y:.4f}
        dy/du = 3u² = {dy_du:.4f}
        
        Chain Rule: dy/dx = dy/du × du/dx
        dy/dx = {dy_du:.4f} × {du_dx:.4f}
        dy/dx = {dy_dx:.4f}
        
        Analytical: {analytical:.4f}
        Match: {np.isclose(dy_dx, analytical)}
        """
        
        ax2.text(0.1, 0.5, text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(demonstrate_chain_rule,
            x_val=FloatSlider(min=-2, max=2, step=0.1, value=1.0,
                             description='x value:',
                             style={'description_width': 'initial'}))


def backprop_widget():
    """
    Interactive widget to step through backpropagation.
    """
    def demonstrate_backprop(x1_val, x2_val):
        from .computation_graph import create_simple_graph
        
        # Create graph: y = (x1 + x2)^2
        graph = create_simple_graph(x1_val, x2_val)
        
        # Forward pass
        y_val = graph.forward()
        
        # Backward pass
        gradients = graph.backward()
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Forward pass visualization
        ax1.text(0.5, 0.9, 'Forward Pass', ha='center', fontsize=14,
                fontweight='bold', transform=ax1.transAxes)
        
        forward_text = f"""
        Inputs:
          x₁ = {x1_val:.2f}
          x₂ = {x2_val:.2f}
        
        Intermediate:
          z = x₁ + x₂ = {x1_val + x2_val:.2f}
        
        Output:
          y = z² = {y_val:.2f}
        """
        
        ax1.text(0.1, 0.5, forward_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax1.transAxes)
        ax1.axis('off')
        
        # Backward pass visualization
        ax2.text(0.5, 0.9, 'Backward Pass', ha='center', fontsize=14,
                fontweight='bold', transform=ax2.transAxes)
        
        z_val = x1_val + x2_val
        backward_text = f"""
        Output gradient:
          ∂y/∂y = 1.0
        
        Intermediate gradient:
          ∂y/∂z = 2z = {2*z_val:.2f}
        
        Input gradients:
          ∂y/∂x₁ = ∂y/∂z × ∂z/∂x₁
                = {2*z_val:.2f} × 1
                = {gradients['x1']:.2f}
          
          ∂y/∂x₂ = ∂y/∂z × ∂z/∂x₂
                = {2*z_val:.2f} × 1
                = {gradients['x2']:.2f}
        """
        
        ax2.text(0.1, 0.5, backward_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(demonstrate_backprop,
            x1_val=FloatSlider(min=-3, max=3, step=0.5, value=2.0,
                              description='x₁:',
                              style={'description_width': 'initial'}),
            x2_val=FloatSlider(min=-3, max=3, step=0.5, value=3.0,
                              description='x₂:',
                              style={'description_width': 'initial'}))


def neural_network_widget():
    """
    Interactive widget for training a simple neural network.
    """
    def train_network(learning_rate, n_epochs):
        from .neural_network import create_simple_network
        
        # Generate simple dataset: XOR problem
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
        Y = np.array([[0, 1, 1, 0]])  # XOR
        
        # Create network
        network = create_simple_network(n_in=2, n_hidden=4, n_out=1)
        
        # Train
        loss_history = network.train(X, Y, epochs=n_epochs, 
                                     learning_rate=learning_rate, verbose=False)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        ax1.plot(loss_history, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Predictions
        predictions = network.predict(X)
        
        ax2.text(0.5, 0.9, 'XOR Problem Results', ha='center', fontsize=14,
                fontweight='bold', transform=ax2.transAxes)
        
        results_text = f"""
        Input  →  Target  Prediction
        [0, 0] →    0      {predictions[0, 0]:.4f}
        [0, 1] →    1      {predictions[0, 1]:.4f}
        [1, 0] →    1      {predictions[0, 2]:.4f}
        [1, 1] →    0      {predictions[0, 3]:.4f}
        
        Final Loss: {loss_history[-1]:.6f}
        """
        
        ax2.text(0.1, 0.5, results_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    interact(train_network,
            learning_rate=FloatSlider(min=0.01, max=1.0, step=0.05, value=0.5,
                                     description='Learning Rate:',
                                     style={'description_width': 'initial'}),
            n_epochs=IntSlider(min=100, max=2000, step=100, value=1000,
                              description='Epochs:',
                              style={'description_width': 'initial'}))
