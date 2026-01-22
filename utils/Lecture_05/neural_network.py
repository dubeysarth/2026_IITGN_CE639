"""
Simple neural network implementation with backpropagation.

This module provides educational implementations of neural network
components for demonstrating backpropagation.
"""

import numpy as np


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        σ(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        σ'(z) = σ(z) * (1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU activation function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        max(0, z)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        1 if z > 0, else 0
    """
    return (z > 0).astype(float)


def tanh(z):
    """
    Tanh activation function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        tanh(z)
    """
    return np.tanh(z)


def tanh_derivative(z):
    """
    Derivative of tanh function.
    
    Parameters:
    -----------
    z : array-like
        Input
    
    Returns:
    --------
    array
        1 - tanh²(z)
    """
    t = np.tanh(z)
    return 1 - t**2


class Layer:
    """
    Single layer in a neural network.
    
    Attributes:
    -----------
    W : array
        Weight matrix (n_out × n_in)
    b : array
        Bias vector (n_out,)
    activation : callable
        Activation function
    activation_derivative : callable
        Derivative of activation function
    """
    
    def __init__(self, n_in, n_out, activation='sigmoid'):
        """
        Initialize a neural network layer.
        
        Parameters:
        -----------
        n_in : int
            Number of input features
        n_out : int
            Number of output features
        activation : str
            Activation function ('sigmoid', 'relu', 'tanh', 'linear')
        """
        # Initialize weights with small random values
        self.W = np.random.randn(n_out, n_in) * 0.1
        self.b = np.zeros((n_out, 1))
        
        # Set activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'linear':
            self.activation = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Cache for backpropagation
        self.z = None  # Pre-activation
        self.a = None  # Post-activation
        self.x = None  # Input
        
        # Gradients
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """
        Forward pass through layer.
        
        Parameters:
        -----------
        x : array
            Input (n_in × batch_size)
        
        Returns:
        --------
        array
            Output (n_out × batch_size)
        """
        self.x = x
        self.z = self.W @ x + self.b
        self.a = self.activation(self.z)
        return self.a
    
    def backward(self, da):
        """
        Backward pass through layer.
        
        Parameters:
        -----------
        da : array
            Gradient of loss w.r.t. activation (n_out × batch_size)
        
        Returns:
        --------
        array
            Gradient of loss w.r.t. input (n_in × batch_size)
        """
        batch_size = self.x.shape[1]
        
        # Gradient through activation
        dz = da * self.activation_derivative(self.z)
        
        # Gradients for parameters
        self.dW = (1 / batch_size) * (dz @ self.x.T)
        self.db = (1 / batch_size) * np.sum(dz, axis=1, keepdims=True)
        
        # Gradient for input
        dx = self.W.T @ dz
        
        return dx
    
    def update(self, learning_rate):
        """
        Update parameters using gradient descent.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class NeuralNetwork:
    """
    Simple feedforward neural network.
    
    Attributes:
    -----------
    layers : list
        List of Layer objects
    """
    
    def __init__(self, layer_sizes, activations=None):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        layer_sizes : list
            List of layer sizes [n_in, n_h1, n_h2, ..., n_out]
        activations : list, optional
            List of activation functions for each layer
        """
        self.layers = []
        
        if activations is None:
            # Default: sigmoid for hidden layers, linear for output
            activations = ['sigmoid'] * (len(layer_sizes) - 2) + ['linear']
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        # Training history
        self.loss_history = []
    
    def forward(self, x):
        """
        Forward pass through network.
        
        Parameters:
        -----------
        x : array
            Input (n_in × batch_size)
        
        Returns:
        --------
        array
            Output (n_out × batch_size)
        """
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a
    
    def backward(self, y_true, y_pred):
        """
        Backward pass through network.
        
        Parameters:
        -----------
        y_true : array
            True labels (n_out × batch_size)
        y_pred : array
            Predicted labels (n_out × batch_size)
        
        Returns:
        --------
        None
        """
        # Gradient of loss w.r.t. output (MSE loss)
        da = y_pred - y_true
        
        # Backpropagate through layers
        for layer in reversed(self.layers):
            da = layer.backward(da)
    
    def update(self, learning_rate):
        """
        Update all parameters.
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate
        """
        for layer in self.layers:
            layer.update(learning_rate)
    
    def train_step(self, x, y, learning_rate=0.01):
        """
        Single training step.
        
        Parameters:
        -----------
        x : array
            Input batch
        y : array
            Target batch
        learning_rate : float
            Learning rate
        
        Returns:
        --------
        float
            Loss value
        """
        # Forward pass
        y_pred = self.forward(x)
        
        # Compute loss (MSE)
        loss = np.mean((y_pred - y)**2)
        
        # Backward pass
        self.backward(y, y_pred)
        
        # Update parameters
        self.update(learning_rate)
        
        return loss
    
    def train(self, X, Y, epochs=100, learning_rate=0.01, verbose=True):
        """
        Train the network.
        
        Parameters:
        -----------
        X : array
            Training inputs (n_in × n_samples)
        Y : array
            Training targets (n_out × n_samples)
        epochs : int
            Number of epochs
        learning_rate : float
            Learning rate
        verbose : bool
            Print progress
        
        Returns:
        --------
        list
            Loss history
        """
        self.loss_history = []
        
        for epoch in range(epochs):
            loss = self.train_step(X, Y, learning_rate)
            self.loss_history.append(loss)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")
        
        return self.loss_history
    
    def predict(self, x):
        """
        Make predictions.
        
        Parameters:
        -----------
        x : array
            Input (n_in × batch_size)
        
        Returns:
        --------
        array
            Predictions (n_out × batch_size)
        """
        return self.forward(x)
    
    def get_gradients(self):
        """
        Get all gradients for inspection.
        
        Returns:
        --------
        list
            List of (dW, db) tuples for each layer
        """
        return [(layer.dW, layer.db) for layer in self.layers]


def create_simple_network(n_in=2, n_hidden=3, n_out=1):
    """
    Create a simple neural network for demonstration.
    
    Parameters:
    -----------
    n_in : int
        Number of inputs
    n_hidden : int
        Number of hidden units
    n_out : int
        Number of outputs
    
    Returns:
    --------
    NeuralNetwork
        Initialized network
    """
    return NeuralNetwork([n_in, n_hidden, n_out], activations=['sigmoid', 'linear'])


def demonstrate_backprop_step_by_step(x, y, network):
    """
    Demonstrate backpropagation step by step.
    
    Parameters:
    -----------
    x : array
        Input
    y : array
        Target
    network : NeuralNetwork
        Network to demonstrate
    
    Returns:
    --------
    dict
        Detailed information about forward and backward passes
    """
    # Forward pass
    activations = [x]
    pre_activations = []
    
    a = x
    for layer in network.layers:
        z = layer.W @ a + layer.b
        a = layer.activation(z)
        pre_activations.append(z)
        activations.append(a)
    
    y_pred = activations[-1]
    loss = np.mean((y_pred - y)**2)
    
    # Backward pass
    gradients = []
    da = y_pred - y
    
    for i, layer in enumerate(reversed(network.layers)):
        layer_idx = len(network.layers) - 1 - i
        
        # Gradient through activation
        dz = da * layer.activation_derivative(pre_activations[layer_idx])
        
        # Parameter gradients
        dW = dz @ activations[layer_idx].T
        db = np.sum(dz, axis=1, keepdims=True)
        
        # Input gradient
        dx = layer.W.T @ dz
        
        gradients.append({
            'layer': layer_idx,
            'dW': dW,
            'db': db,
            'dz': dz,
            'da': da
        })
        
        da = dx
    
    return {
        'forward': {
            'activations': activations,
            'pre_activations': pre_activations,
            'prediction': y_pred,
            'loss': loss
        },
        'backward': {
            'gradients': list(reversed(gradients))
        }
    }
