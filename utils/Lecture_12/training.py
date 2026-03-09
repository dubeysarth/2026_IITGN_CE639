"""
training.py — Sequence Data Pipeline and Training Loop
=======================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides everything needed to train RNN/LSTM/GRU models on time-series data:
  • create_sequences         — sliding window windowing
  • temporal_train_val_split — chronological split (NEVER shuffle!)
  • SequenceDataset          — PyTorch Dataset
  • train_one_epoch          — one epoch with gradient clipping
  • evaluate                 — validation / test evaluation
  • train_rnn                — full training loop returning a history dict
  • plot_training_history    — loss & metric curves
  • nse_score                — Nash–Sutcliffe Efficiency (hydrological metric)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, List, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ===========================================================================
# Data Preparation
# ===========================================================================

def create_sequences(
    data      : np.ndarray,
    lookback  : int,
    horizon   : int = 1,
    target_col: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a multivariate time-series into (input, target) sliding windows.

    For each window k:
        X[k] = data[k : k+lookback]          shape: (lookback, n_features)
        y[k] = data[k+lookback : k+lookback+horizon, target_col]

    Parameters
    ----------
    data       : (N, n_features) or (N,) time series
    lookback   : L — context length (input window size)
    horizon    : number of future steps to predict (default=1 → scalar target)
    target_col : which column is the target variable

    Returns
    -------
    X : (N-lookback-horizon+1, lookback, n_features) — input sequences
    y : (N-lookback-horizon+1, horizon)              — target values
    """
    if data.ndim == 1:
        data = data[:, None]

    N, n_features = data.shape
    n_windows = N - lookback - horizon + 1

    if n_windows <= 0:
        raise ValueError(
            f"Not enough data: N={N}, lookback={lookback}, horizon={horizon}. "
            f"Need N > lookback + horizon = {lookback + horizon}."
        )

    X = np.zeros((n_windows, lookback, n_features), dtype=np.float32)
    y = np.zeros((n_windows, horizon),               dtype=np.float32)

    for k in range(n_windows):
        X[k] = data[k : k + lookback]
        y[k] = data[k + lookback : k + lookback + horizon, target_col]

    if horizon == 1:
        y = y[:, 0]   # squeeze to (n_windows,) for scalar regression

    return X, y


def temporal_train_val_split(
    X      : np.ndarray,
    y      : np.ndarray,
    val_frac: float = 0.2,
    test_frac: float = 0.0
) -> Tuple:
    """
    Chronological (non-shuffling) train / validation / test split.

    IMPORTANT: Never shuffle a time series across split boundaries — future
    observations must not leak into training.

    Parameters
    ----------
    X         : (N, lookback, d)
    y         : (N,) or (N, horizon)
    val_frac  : fraction for validation (taken from the end of train)
    test_frac : fraction for test (taken after validation)

    Returns
    -------
    (X_train, y_train, X_val, y_val)
    or if test_frac > 0:
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    N = len(X)
    n_test = int(N * test_frac)
    n_val  = int(N * val_frac)
    n_train = N - n_val - n_test

    X_train, y_train = X[:n_train],            y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]

    if test_frac > 0:
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_val, y_val


class SequenceDataset(Dataset if TORCH_AVAILABLE else object):
    """
    PyTorch Dataset wrapping (X, y) arrays of windowed sequences.

    Converts NumPy arrays to float32 tensors on first access.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SequenceDataset")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===========================================================================
# Training Utilities
# ===========================================================================

def train_one_epoch(
    model       : 'nn.Module',
    loader      : 'DataLoader',
    optimizer   : 'torch.optim.Optimizer',
    criterion   : 'nn.Module',
    clip_value  : float = 5.0,
    device      : str = 'cpu'
) -> Tuple[float, List[float]]:
    """
    Train the model for one epoch with gradient clipping.

    Parameters
    ----------
    model      : PyTorch model
    loader     : training DataLoader
    optimizer  : e.g. Adam
    criterion  : loss function (MSELoss, CrossEntropyLoss)
    clip_value : max gradient norm for clipping (lecture default: 5.0)
    device     : 'cpu' or 'cuda'

    Returns
    -------
    epoch_loss  : average loss over all batches
    grad_norms  : list of gradient norms before clipping (for monitoring)
    """
    model.train()
    total_loss = 0.0
    grad_norms = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_hat = model(X_batch)

        # Squeeze output if needed (regression: (B,1) → (B,))
        if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze(-1)

        loss = criterion(y_hat, y_batch)
        loss.backward()

        # Compute gradient norm BEFORE clipping (for lecture demos)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        # Clip gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss, grad_norms


def evaluate(
    model    : 'nn.Module',
    loader   : 'DataLoader',
    criterion: 'nn.Module',
    device   : str = 'cpu'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a DataLoader.

    Returns
    -------
    loss    : average loss
    y_true  : ground-truth numpy array
    y_pred  : prediction numpy array
    """
    model.eval()
    total_loss = 0.0
    all_true  = []
    all_pred  = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(X_batch)
            if y_hat.ndim > 1 and y_hat.shape[-1] == 1:
                y_hat = y_hat.squeeze(-1)

            loss = criterion(y_hat, y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_true.append(y_batch.cpu().numpy())
            all_pred.append(y_hat.cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return epoch_loss, y_true, y_pred


def train_rnn(
    model       : 'nn.Module',
    train_loader: 'DataLoader',
    val_loader  : 'DataLoader',
    n_epochs    : int = 30,
    lr          : float = 1e-3,
    clip_value  : float = 5.0,
    device      : str = 'cpu',
    patience    : int = 10,
    verbose     : bool = True
) -> Dict[str, List]:
    """
    Full training loop for RNN/LSTM/GRU models.

    Features:
      • Adam optimiser (better than SGD for RNNs)
      • Gradient clipping (mandatory for vanilla RNNs)
      • Early stopping on validation loss
      • Returns complete history dict for plotting

    Parameters
    ----------
    model        : PyTorch model
    train_loader : DataLoader for training set
    val_loader   : DataLoader for validation set
    n_epochs     : maximum training epochs
    lr           : learning rate (1e-3 default is reliable for LSTM)
    clip_value   : gradient clipping threshold
    device       : 'cpu' or 'cuda'
    patience     : early stopping patience
    verbose      : print epoch summary

    Returns
    -------
    history : dict with lists:
        'train_loss', 'val_loss', 'grad_norms', 'epoch'
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss'  : [],
        'grad_norms': [],
        'epoch'     : []
    }

    best_val  = float('inf')
    best_state= None
    no_improve= 0

    for epoch in range(1, n_epochs + 1):
        train_loss, g_norms = train_one_epoch(
            model, train_loader, optimizer, criterion, clip_value, device
        )
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['grad_norms'].append(np.mean(g_norms))
        history['epoch'].append(epoch)

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val    = val_loss
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.5f} | "
                  f"Val Loss: {val_loss:.5f} | "
                  f"Grad Norm: {np.mean(g_norms):.3f}")

        if no_improve >= patience:
            if verbose:
                print(f"  ⚡ Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n  ✅ Training complete. Best val loss: {best_val:.5f}")

    return history


# ===========================================================================
# Metrics
# ===========================================================================

def nse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash–Sutcliffe Efficiency (NSE).

    NSE = 1 - Σ(Q_obs - Q_pred)² / Σ(Q_obs - Q̄_obs)²

    Interpretation:
        NSE = 1.0  → perfect prediction
        NSE > 0.7  → "good" (standard hydrology threshold)
        NSE = 0.0  → model performs same as mean
        NSE < 0.0  → model worse than simple mean

    References: Nash & Sutcliffe (1970), Kratzert et al. (2019)
    """
    mean_obs = np.mean(y_true)
    numer    = np.sum((y_true - y_pred) ** 2)
    denom    = np.sum((y_true - mean_obs) ** 2)
    if denom < 1e-12:
        return float('nan')
    return float(1.0 - numer / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_training_history(
    history  : Dict[str, List],
    figsize  : Tuple[int, int] = (14, 5),
    show_grad: bool = True
) -> Tuple:
    """
    Plot training / validation loss (and optionally gradient norms).

    Parameters
    ----------
    history   : dict from train_rnn()
    figsize   : figure dimensions
    show_grad : whether to plot average gradient norm per epoch

    Returns
    -------
    (fig, axes)
    """
    n_plots = 3 if show_grad else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    epochs = history['epoch']

    # --- Loss ---
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-o', markersize=3,
            linewidth=2, label='Train Loss')
    ax.plot(epochs, history['val_loss'],   'r-o', markersize=3,
            linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Val Loss zoomed ---
    ax = axes[1]
    ax.plot(epochs, history['val_loss'], 'r-o', markersize=3, linewidth=2)
    best_epoch = np.argmin(history['val_loss'])
    ax.axvline(epochs[best_epoch], color='green', linestyle='--', linewidth=1.5,
               label=f'Best (epoch {epochs[best_epoch]})')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val MSE', fontsize=11)
    ax.set_title('Validation Loss Detail', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Gradient norms ---
    if show_grad and 'grad_norms' in history:
        ax = axes[2]
        ax.plot(epochs, history['grad_norms'], 'g-o', markersize=3, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Avg Grad Norm', fontsize=11)
        ax.set_title('Gradient Norm (avg per epoch)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("Testing training utilities...")

    # Test create_sequences
    np.random.seed(42)
    N, d = 100, 3
    data = np.random.randn(N, d).astype(np.float32)
    X, y = create_sequences(data, lookback=10, horizon=1, target_col=0)
    print(f"  create_sequences: X{X.shape}, y{y.shape}")
    assert X.shape == (90, 10, 3), f"X shape wrong: {X.shape}"
    assert y.shape == (90,),       f"y shape wrong: {y.shape}"

    # Multi-step
    X_ms, y_ms = create_sequences(data, lookback=10, horizon=5, target_col=0)
    assert X_ms.shape == (86, 10, 3)
    assert y_ms.shape == (86, 5)
    print(f"  Multi-step: X{X_ms.shape}, y{y_ms.shape}")

    # Temporal split
    X_tr, y_tr, X_val, y_val = temporal_train_val_split(X, y, val_frac=0.2)
    print(f"  Temporal split: train={len(X_tr)}, val={len(X_val)}")
    assert len(X_tr) + len(X_val) == 90

    # NSE score
    y_true  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred  = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    nse_val = nse_score(y_true, y_pred)
    print(f"  NSE score (near-perfect pred): {nse_val:.4f}  (expected ≈ 0.99)")

    # PyTorch pipeline
    if TORCH_AVAILABLE:
        from utils.Lecture_12.architectures import SimpleLSTM
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

        ds_tr  = SequenceDataset(X_tr, y_tr)
        ds_val = SequenceDataset(X_val, y_val)
        dl_tr  = DataLoader(ds_tr,  batch_size=16, shuffle=False)
        dl_val = DataLoader(ds_val, batch_size=16, shuffle=False)

        model = SimpleLSTM(input_size=3, hidden_size=16, num_layers=1, output_size=1)
        history = train_rnn(model, dl_tr, dl_val, n_epochs=5, verbose=True)
        print(f"  Final train loss: {history['train_loss'][-1]:.5f}")
    else:
        print("  ⚠️ PyTorch not available — skipping model training test")

    print("\n✅ training.py: all self-tests passed!")
