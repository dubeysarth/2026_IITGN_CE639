"""
Autoencoders for Dimensionality Reduction

This module implements simple autoencoders from scratch:
1. LinearAutoencoder - Equivalent to PCA
2. Autoencoder - Nonlinear with activation functions
"""

import numpy as np


class LinearAutoencoder:
    """
    Linear autoencoder (equivalent to PCA).
    
    Architecture:
    - Encoder: X @ W_e^T → Z (latent)
    - Decoder: Z @ W_d^T → X_reconstructed
    
    With linear activations and MSE loss, this is mathematically
    equivalent to PCA.
    """
    
    def __init__(self, input_dim, latent_dim, random_state=None):
        """
        Initialize linear autoencoder.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        latent_dim : int
            Dimension of latent space
        random_state : int, optional
            Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Initialize weights (small random values)
        self.W_encoder = np.random.randn(latent_dim, input_dim) * 0.01
        self.W_decoder = np.random.randn(input_dim, latent_dim) * 0.01
        
        self.b_encoder = np.zeros(latent_dim)
        self.b_decoder = np.zeros(input_dim)
    
    def encode(self, X):
        """Encode to latent space."""
        return X @ self.W_encoder.T + self.b_encoder
    
    def decode(self, Z):
        """Decode from latent space."""
        return Z @ self.W_decoder.T + self.b_decoder
    
    def forward(self, X):
        """Forward pass: encode then decode."""
        Z = self.encode(X)
        X_reconstructed = self.decode(Z)
        return X_reconstructed, Z
    
    def compute_loss(self, X, X_reconstructed):
        """MSE reconstruction loss."""
        return np.mean((X - X_reconstructed) ** 2)
    
    def backward(self, X, X_reconstructed, Z, learning_rate=0.01):
        """
        Backpropagation for linear autoencoder.
        
        Gradients:
        dL/dW_d = (X_reconstructed - X)^T @ Z
        dL/dW_e = W_d^T @ (X_reconstructed - X)^T @ X
        """
        n_samples = X.shape[0]
        
        # Error
        error = X_reconstructed - X  # (n, input_dim)
        
        # Decoder gradients
        dW_decoder = (error.T @ Z) / n_samples
        db_decoder = np.mean(error, axis=0)
        
        # Encoder gradients (backprop through decoder)
        error_latent = error @ self.W_decoder  # (n, latent_dim)
        dW_encoder = (error_latent.T @ X) / n_samples
        db_encoder = np.mean(error_latent, axis=0)
        
        # Update weights
        self.W_decoder -= learning_rate * dW_decoder
        self.b_decoder -= learning_rate * db_decoder
        self.W_encoder -= learning_rate * dW_encoder
        self.b_encoder -= learning_rate * db_encoder


class Autoencoder:
    """
    Nonlinear autoencoder with activation functions.
    
    Architecture:
    - Encoder: X → Hidden → Latent (with activations)
    - Decoder: Latent → Hidden → Output (with activations)
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, random_state=None):
        """
        Initialize nonlinear autoencoder.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dim : int
            Hidden layer size
        latent_dim : int
            Latent space dimension
        random_state : int, optional
            Random seed
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder weights
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(latent_dim)
        
        # Decoder weights
        self.W3 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / latent_dim)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b4 = np.zeros(input_dim)
        
        # Cache for backprop
        self.cache = {}
    
    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)
    
    def encode(self, X):
        """Encode to latent space."""
        h1 = self.relu(X @ self.W1.T + self.b1)
        z = X @ self.W2.T + self.b2  # Linear for latent
        return z
    
    def decode(self, Z):
        """Decode from latent space."""
        h2 = self.relu(Z @ self.W3.T + self.b3)
        x_out = h2 @ self.W4.T + self.b4  # Linear output
        return x_out
    
    def forward(self, X):
        """Forward pass with caching for backprop."""
        # Encoder
        z1 = X @ self.W1.T + self.b1
        h1 = self.relu(z1)
        z2 = h1 @ self.W2.T + self.b2
        latent = z2  # Linear latent
        
        # Decoder
        z3 = latent @ self.W3.T + self.b3
        h2 = self.relu(z3)
        z4 = h2 @ self.W4.T + self.b4
        output = z4  # Linear output
        
        # Cache
        self.cache = {
            'X': X, 'z1': z1, 'h1': h1, 'z2': z2, 'latent': latent,
            'z3': z3, 'h2': h2, 'z4': z4, 'output': output
        }
        
        return output, latent
    
    def compute_loss(self, X, X_reconstructed):
        """MSE reconstruction loss."""
        return np.mean((X - X_reconstructed) ** 2)
    
    def backward(self, learning_rate=0.001):
        """
        Backpropagation through the autoencoder.
        """
        n_samples = self.cache['X'].shape[0]
        
        # Output error
        dz4 = self.cache['output'] - self.cache['X']
        
        # Decoder layer 2
        dW4 = (dz4.T @ self.cache['h2']) / n_samples
        db4 = np.mean(dz4, axis=0)
        dh2 = dz4 @ self.W4
        
        # Decoder layer 1
        dz3 = dh2 * self.relu_derivative(self.cache['z3'])
        dW3 = (dz3.T @ self.cache['latent']) / n_samples
        db3 = np.mean(dz3, axis=0)
        dlatent = dz3 @ self.W3
        
        # Encoder layer 2
        dz2 = dlatent
        dW2 = (dz2.T @ self.cache['h1']) / n_samples
        db2 = np.mean(dz2, axis=0)
        dh1 = dz2 @ self.W2
        
        # Encoder layer 1
        dz1 = dh1 * self.relu_derivative(self.cache['z1'])
        dW1 = (dz1.T @ self.cache['X']) / n_samples
        db1 = np.mean(dz1, axis=0)
        
        # Update weights
        self.W4 -= learning_rate * dW4
        self.b4 -= learning_rate * db4
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


def train_autoencoder(autoencoder, X, epochs=100, learning_rate=0.01, 
                     batch_size=None, verbose=True):
    """
    Train an autoencoder.
    
    Parameters:
    -----------
    autoencoder : LinearAutoencoder or Autoencoder
        Autoencoder instance
    X : array_like
        Training data
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    batch_size : int, optional
        Batch size (None = full batch)
    verbose : bool
        Print progress
    
    Returns:
    --------
    dict
        Training history with loss per epoch
    """
    n_samples = X.shape[0]
    
    if batch_size is None:
        batch_size = n_samples
    
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        
        epoch_loss = 0
        n_batches = 0
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            
            # Forward pass
            X_reconstructed, Z = autoencoder.forward(X_batch)
            
            # Compute loss
            loss = autoencoder.compute_loss(X_batch, X_reconstructed)
            epoch_loss += loss
            n_batches += 1
            
            # Backward pass
            if isinstance(autoencoder, LinearAutoencoder):
                autoencoder.backward(X_batch, X_reconstructed, Z, learning_rate)
            else:
                autoencoder.backward(learning_rate)
        
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return {'loss_history': loss_history}


def encode(autoencoder, X):
    """Encode data to latent space."""
    return autoencoder.encode(X)


def decode(autoencoder, Z):
    """Decode from latent space."""
    return autoencoder.decode(Z)


def reconstruction_error(autoencoder, X):
    """
    Compute reconstruction error.
    
    Parameters:
    -----------
    autoencoder : LinearAutoencoder or Autoencoder
        Trained autoencoder
    X : array_like
        Data to reconstruct
    
    Returns:
    --------
    dict
        Contains:
        - 'mse': Mean squared error
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
    """
    X_reconstructed, _ = autoencoder.forward(X)
    
    mse = np.mean((X - X_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(X - X_reconstructed))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'X_reconstructed': X_reconstructed
    }
