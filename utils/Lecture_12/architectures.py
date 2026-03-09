"""
architectures.py — PyTorch RNN/LSTM/GRU/BiLSTM Model Classes
=============================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides ready-to-train PyTorch model classes for all recurrent architectures
discussed in the lecture, plus utilities for parameter counting and model info.

All models follow the same interface:
    model(x)   where x : (batch, seq_len, input_size)
    returns    y_hat : (batch, output_size)   [many-to-one]
    or         y_hat : (batch, seq_len, output_size) [many-to-many]

Requires PyTorch (pip install torch).  The module can be imported without PyTorch;
model instantiation will raise a clear ImportError.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
    TORCH_AVAILABLE = True
except ImportError:
    torch = None          # type: ignore
    nn    = None          # type: ignore
    _Module = object      # safe fallback base so class definitions don't fail
    TORCH_AVAILABLE = False


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for architectures.py.\n"
            "Install with: pip install torch"
        )


# ===========================================================================
# Model Classes
# ===========================================================================

class SimpleRNN(_Module):
    """
    Vanilla Elman RNN for sequence regression / classification.

    Architecture:
        nn.RNN (stacked, with optional dropout between layers)
        Linear read-out

    Parameters
    ----------
    input_size  : d — number of features per time step
    hidden_size : n_h
    num_layers  : number of stacked RNN layers
    output_size : dimensionality of prediction
    dropout     : inter-layer dropout (only applied if num_layers > 1)
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, output_size: int = 1,
                 dropout: float = 0.0):
        _check_torch()
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.rnn = nn.RNN(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            nonlinearity = 'tanh'
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: 'torch.Tensor',
                h0: Optional['torch.Tensor'] = None) -> 'torch.Tensor':
        """
        x   : (B, T, d)
        h0  : (num_layers, B, n_h)  or None (zeros)
        out : (B, output_size)  — prediction from the final hidden state
        """
        out, _ = self.rnn(x, h0)     # out : (B, T, n_h)
        out     = out[:, -1, :]       # take last time step
        return self.fc(out)


class SimpleLSTM(_Module):
    """
    LSTM model for sequence regression or classification (many-to-one).

    The default CE use-case: predict Q_{t+1} from a window of length L.

    Parameters
    ----------
    input_size  : d
    hidden_size : n_h
    num_layers  : stacked LSTM layers
    output_size : prediction dimension
    dropout     : inter-layer dropout
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, output_size: int = 1,
                 dropout: float = 0.0):
        _check_torch()
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x: 'torch.Tensor',
                state: Optional[Tuple] = None) -> 'torch.Tensor':
        """
        x     : (B, T, d)
        state : optional (h0, c0) tuple
        out   : (B, output_size)
        """
        out, (h_n, _) = self.lstm(x, state)  # out : (B, T, n_h)
        # h_n : (num_layers, B, n_h) — take last layer
        return self.fc(h_n[-1])


class SimpleGRU(_Module):
    """
    GRU model for sequence regression / classification (many-to-one).

    Fewer parameters than LSTM; often comparable performance on CE tasks.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, output_size: int = 1,
                 dropout: float = 0.0):
        _check_torch()
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x: 'torch.Tensor',
                h0: Optional['torch.Tensor'] = None) -> 'torch.Tensor':
        out, h_n = self.gru(x, h0)
        return self.fc(h_n[-1])


class BidirectionalLSTM(_Module):
    """
    Bidirectional LSTM for offline sequence classification (many-to-one).

    Suitable for: seismic event classification, post-event SHM, etc.
    NOT suitable for real-time forecasting (requires future observations).

    The concatenated forward + backward final hidden states are fed to FC.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, output_size: int = 2,
                 dropout: float = 0.0):
        _check_torch()
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0
        )
        # Bidirectional doubles the hidden dim
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        out, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, n_h) — take last layer's fwd & bwd
        h_fwd = h_n[-2]   # forward  final hidden
        h_bwd = h_n[-1]   # backward final hidden
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2*n_h)
        return self.fc(self.dropout(h_cat))


class StackedLSTM(_Module):
    """
    Multi-layer LSTM with proper inter-layer dropout and a configurable head.

    PyTorch's nn.LSTM dropout applies between layers (not on recurrent connections),
    which is correct per the lecture's best-practices.

    Parameters
    ----------
    input_size  : d
    hidden_size : n_h per layer
    num_layers  : 1–3 recommended for CE tasks
    output_size : prediction dimension
    dropout     : inter-layer dropout probability
    """

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 2, output_size: int = 1,
                 dropout: float = 0.3):
        _check_torch()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        out, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class Seq2SeqLSTM(_Module):
    """
    Encoder-Decoder LSTM for multi-step sequence forecasting (many-to-many).

    Architecture:
        Encoder: processes input sequence → context vector (h_n, c_n)
        Decoder: generates output sequence step-by-step

    Usage: traffic multi-step forecasting, multi-day streamflow forecast.

    Parameters
    ----------
    input_size    : d — encoder input features
    hidden_size   : n_h
    output_size   : d_out — decoder output features per step
    horizon       : number of future steps to predict
    num_layers    : stacked LSTM layers
    dropout       : inter-layer dropout
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = 1, horizon: int = 7,
                 num_layers: int = 1, dropout: float = 0.0):
        _check_torch()
        super().__init__()
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.LSTM(
            input_size  = output_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        x   : (B, T, d)
        out : (B, horizon, output_size)
        """
        B = x.size(0)

        # Encode input sequence
        _, (h_n, c_n) = self.encoder(x)

        # Decoder: start token = zeros; autoregressively produce horizon steps
        dec_input = torch.zeros(B, 1, self.fc.out_features,
                                device=x.device, dtype=x.dtype)
        state = (h_n, c_n)
        outputs = []

        for _ in range(self.horizon):
            dec_out, state = self.decoder(dec_input, state)  # (B,1,n_h)
            pred            = self.fc(dec_out)                # (B,1,out_size)
            outputs.append(pred)
            dec_input = pred   # feed prediction back as next input

        return torch.cat(outputs, dim=1)   # (B, horizon, output_size)


# ===========================================================================
# Utilities
# ===========================================================================

def count_parameters(model: 'nn.Module') -> Dict[str, int]:
    """
    Count total and trainable parameters in a PyTorch model.

    Returns dict: {'total': int, 'trainable': int}
    """
    _check_torch()
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def model_summary(model: 'nn.Module',
                  input_shape: Optional[Tuple[int, int, int]] = None) -> None:
    """
    Print a concise layer-by-layer summary of a recurrent model.

    Parameters
    ----------
    model       : PyTorch nn.Module
    input_shape : (batch, seq_len, input_size) — if provided, do a dry run
    """
    _check_torch()
    print("=" * 65)
    print(f"  Model: {model.__class__.__name__}")
    print("=" * 65)
    print(f"  {'Layer':<30} {'Parameters':>12}")
    print("-" * 65)

    total = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        print(f"  {name:<30} {n:>12,}")

    print("-" * 65)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {'TOTAL':<30} {total:>12,}")
    print(f"  {'Trainable':<30} {trainable:>12,}")
    print("=" * 65)

    if input_shape is not None:
        try:
            x = torch.zeros(*input_shape)
            with torch.no_grad():
                out = model(x)
            print(f"  Input shape:  {tuple(x.shape)}")
            print(f"  Output shape: {tuple(out.shape)}")
            print("=" * 65)
        except Exception as e:
            print(f"  Forward pass failed: {e}")


def compare_architectures(d: int = 3, n_h: int = 64, n_out: int = 1) -> None:
    """
    Print a comparison table of RNN/LSTM/GRU/BiLSTM parameter counts.
    """
    _check_torch()
    models = {
        'SimpleRNN(1 layer)' : SimpleRNN(d, n_h, 1, n_out),
        'SimpleRNN(2 layers)': SimpleRNN(d, n_h, 2, n_out),
        'SimpleLSTM(1 layer)': SimpleLSTM(d, n_h, 1, n_out),
        'SimpleLSTM(2 layers)': SimpleLSTM(d, n_h, 2, n_out),
        'SimpleGRU(1 layer)' : SimpleGRU(d, n_h, 1, n_out),
        'BiLSTM(1 layer)'    : BidirectionalLSTM(d, n_h, 1, 2),
    }

    print("\n" + "=" * 55)
    print(f"  Architecture Comparison  (d={d}, n_h={n_h})")
    print("=" * 55)
    print(f"  {'Architecture':<25} {'Parameters':>12}")
    print("-" * 55)
    for name, m in models.items():
        p = count_parameters(m)
        print(f"  {name:<25} {p['total']:>12,}")
    print("=" * 55)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    _check_torch()
    import torch
    torch.manual_seed(42)

    B, T, d, n_h = 4, 30, 3, 32

    print("Testing SimpleRNN...")
    m = SimpleRNN(d, n_h, num_layers=1, output_size=1)
    x = torch.randn(B, T, d)
    out = m(x)
    assert out.shape == (B, 1), f"RNN output shape: {out.shape}"
    print(f"  ✓ Output shape: {out.shape}")

    print("\nTesting SimpleLSTM...")
    m = SimpleLSTM(d, n_h, num_layers=2, output_size=1, dropout=0.3)
    out = m(x)
    assert out.shape == (B, 1)
    print(f"  ✓ Output shape: {out.shape}")

    print("\nTesting SimpleGRU...")
    m = SimpleGRU(d, n_h, output_size=1)
    out = m(x)
    assert out.shape == (B, 1)
    print(f"  ✓ Output shape: {out.shape}")

    print("\nTesting BidirectionalLSTM...")
    m = BidirectionalLSTM(d, n_h, output_size=2)
    out = m(x)
    assert out.shape == (B, 2)
    print(f"  ✓ Output shape: {out.shape}")

    print("\nTesting StackedLSTM...")
    m = StackedLSTM(d, n_h, num_layers=2, output_size=1, dropout=0.2)
    out = m(x)
    assert out.shape == (B, 1)
    print(f"  ✓ Output shape: {out.shape}")
    model_summary(m, input_shape=(B, T, d))

    print("\nTesting Seq2SeqLSTM...")
    horizon = 7
    m = Seq2SeqLSTM(d, n_h, output_size=1, horizon=horizon)
    out = m(x)
    assert out.shape == (B, horizon, 1), f"{out.shape}"
    print(f"  ✓ Output shape: {out.shape}")

    compare_architectures(d=3, n_h=64)
    print("\n✅ architectures.py: all self-tests passed!")
