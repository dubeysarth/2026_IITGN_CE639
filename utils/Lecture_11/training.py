"""
Training utilities for CNNs.

Provides training loops, evaluation, and history tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List, Optional


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: str = 'cpu') -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    device : str
        Device to train on
        
    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader,
            criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluate model on validation/test set.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    loader : DataLoader
        Validation/test data loader
    criterion : nn.Module
        Loss function
    device : str
        Device
        
    Returns
    -------
    Tuple[float, float]
        (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_cnn(model: nn.Module, train_loader: DataLoader,
              val_loader: DataLoader, epochs: int = 10,
              lr: float = 0.001, device: str = 'cpu',
              verbose: bool = True) -> Dict:
    """
    Full training loop with history tracking.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    epochs : int
        Number of epochs
    lr : float
        Learning rate
    device : str
        Device
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict
        Training history with keys:
        - 'train_loss', 'train_acc'
        - 'val_loss', 'val_acc'
        - 'model' (trained model)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    history['model'] = model
    return history


def plot_training_history(history: Dict, figsize: Tuple[int, int] = (14, 5)):
    """
    Plot training and validation curves.
    
    Parameters
    ----------
    history : Dict
        Training history from train_cnn()
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes


def get_predictions(model: nn.Module, loader: DataLoader,
                   device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions and ground truth labels.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    loader : DataLoader
        Data loader
    device : str
        Device
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (predictions, ground_truth)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_labels)


if __name__ == "__main__":
    print("Training utilities module loaded successfully!")
    print("Note: Actual training requires PyTorch and data loaders")
