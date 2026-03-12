"""
PyTorch FNN architectures for Lecture 10: Feedforward Neural Networks.

Provides configurable MLP implementations, CE-specific network designs,
a factory function, and parameter/summary utilities.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Configurable MLP
# ─────────────────────────────────────────────────────────────────────────────

def _build_pytorch_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> "torch.nn.Sequential":
    """
    Internal helper: construct a Sequential MLP with given configuration.

    Parameters
    ----------
    input_dim : int
    hidden_dims : List[int]
    output_dim : int
    activation : str
    dropout : float
    batch_norm : bool

    Returns
    -------
    torch.nn.Sequential
    """
    import torch.nn as nn

    act_map = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
    }
    if activation not in act_map:
        raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(act_map.keys())}")

    layers = []
    dims = [input_dim] + hidden_dims

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        layers.append(type(act_map[activation])())   # fresh instance each time
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

    # Output layer (no activation — applied by loss or task)
    layers.append(nn.Linear(dims[-1], output_dim))

    return nn.Sequential(*layers)


class SimpleFNN:
    """
    Configurable multi-layer perceptron (PyTorch).

    Wraps torch.nn.Sequential with configuration tracking for easy
    parameter counting and summaries.

    Usage
    -----
    model = SimpleFNN(input_dim=6, hidden_dims=[32, 16], output_dim=1)
    net = model.build()       # returns a torch.nn.Module
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

    def build(self):
        """Return a torch.nn.Sequential model."""
        return _build_pytorch_layers(
            self.input_dim, self.hidden_dims, self.output_dim,
            self.activation, self.dropout, self.batch_norm
        )

    def __repr__(self):
        return (f"SimpleFNN(input={self.input_dim}, hidden={self.hidden_dims}, "
                f"output={self.output_dim}, act={self.activation}, "
                f"dropout={self.dropout}, bn={self.batch_norm})")


# ─────────────────────────────────────────────────────────────────────────────
# CE-Specific Architectures
# ─────────────────────────────────────────────────────────────────────────────

def BeamDeflectionNet(
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    """
    FNN for predicting RC beam midspan deflection.

    Input features (6):
        P      — Point load [kN]
        L      — Beam length [m]
        E      — Elastic modulus [GPa]
        I_gross — Gross moment of inertia [m^4]
        M_cr   — Cracking moment [kN·m]
        rho    — Reinforcement ratio [-]

    Output (1):
        delta  — Midspan deflection [mm]

    Architecture default:
        Input(6) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1)
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]
    cfg = SimpleFNN(
        input_dim=6, hidden_dims=hidden_dims, output_dim=1,
        activation="relu", dropout=dropout, batch_norm=batch_norm
    )
    return cfg.build()


def ConcreteStrengthNet(
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    """
    FNN for predicting 28-day concrete compressive strength.

    Input features (8):
        cement, water, fine_aggregate, coarse_aggregate,
        fly_ash, slag, superplasticizer, curing_time

    Output (1):
        f_c — 28-day compressive strength [MPa]

    Architecture default:
        Input(8) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(16) → ReLU → Dense(1)
    """
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]
    cfg = SimpleFNN(
        input_dim=8, hidden_dims=hidden_dims, output_dim=1,
        activation="relu", dropout=dropout, batch_norm=batch_norm
    )
    return cfg.build()


def TrafficFlowNet(
    n_classes: int = 3,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.1,
):
    """
    FNN for classifying traffic flow conditions.

    Input features (5):
        volume_upstream, speed_mean, occupancy, time_of_day_sin, time_of_day_cos

    Output (n_classes):
        Traffic condition class (Free, Congested, HeavilyCongested)
    """
    if hidden_dims is None:
        hidden_dims = [32, 16]
    cfg = SimpleFNN(
        input_dim=5, hidden_dims=hidden_dims, output_dim=n_classes,
        activation="relu", dropout=dropout
    )
    return cfg.build()


# ─────────────────────────────────────────────────────────────────────────────
# Factory from Config Dict
# ─────────────────────────────────────────────────────────────────────────────

def build_fnn(config: Dict):
    """
    Build a SimpleFNN from a configuration dictionary.

    Parameters
    ----------
    config : dict
        Required: 'input_dim', 'hidden_dims', 'output_dim'
        Optional: 'activation', 'dropout', 'batch_norm'

    Returns
    -------
    torch.nn.Sequential

    Example
    -------
    >>> cfg = {"input_dim": 6, "hidden_dims": [32, 16], "output_dim": 1}
    >>> net = build_fnn(cfg)
    """
    return SimpleFNN(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        activation=config.get("activation", "relu"),
        dropout=config.get("dropout", 0.0),
        batch_norm=config.get("batch_norm", False),
    ).build()


# ─────────────────────────────────────────────────────────────────────────────
# Parameter & Summary Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model) -> Dict[str, int]:
    """
    Count trainable and total parameters in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    dict with 'total' and 'trainable' keys
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def model_summary(model, input_size: Tuple[int, ...]) -> str:
    """
    Print a layer-by-layer summary of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
    input_size : Tuple[int, ...]
        Shape of a single input sample (excluding batch dim).

    Returns
    -------
    str — the summary string
    """
    import torch

    try:
        import torchsummary
        # If available use torchsummary
        torchsummary.summary(model, input_size)
        return ""
    except ImportError:
        pass

    lines = ["Model Summary", "=" * 60]
    lines.append(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':>10}")
    lines.append("-" * 60)

    total_params = 0
    x = torch.zeros(1, *input_size)

    for name, layer in model.named_modules():
        if name == "":
            continue
        try:
            y = layer(x)
        except Exception:
            y = x

        params = sum(p.numel() for p in layer.parameters())
        total_params += params
        lines.append(f"{name:<30} {str(tuple(y.shape)):<20} {params:>10,}")

    lines.append("=" * 60)
    lines.append(f"{'Total parameters':>48} {total_params:>10,}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        import torch

        net_beam = BeamDeflectionNet()
        beam_params = count_parameters(net_beam)
        print(f"BeamDeflectionNet: {beam_params}")

        net_concrete = ConcreteStrengthNet()
        conc_params = count_parameters(net_concrete)
        print(f"ConcreteStrengthNet: {conc_params}")

        cfg = {"input_dim": 6, "hidden_dims": [64, 32, 16], "output_dim": 1,
               "activation": "relu", "dropout": 0.1, "batch_norm": True}
        net_factory = build_fnn(cfg)
        print(f"build_fnn result: {count_parameters(net_factory)}")

        print("architectures.py loaded OK ✓")
    except ImportError:
        print("PyTorch not available — architectures.py structural check only.")
        print("architectures.py loaded OK ✓")
