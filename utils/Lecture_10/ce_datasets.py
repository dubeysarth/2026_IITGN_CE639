"""
Synthetic CE datasets for Lecture 10: Feedforward Neural Networks.

All datasets are procedurally generated using physics-inspired formulas so
no external downloads are required. Each function returns (X, y) arrays
with realistic noise and nonlinear structure.

CE 639: AI for Civil Engineering — IIT Gandhinagar
"""

import numpy as np
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# CE Dataset 1 — RC Beam Midspan Deflection
# ─────────────────────────────────────────────────────────────────────────────

def generate_beam_deflection_dataset(
    n: int = 1000,
    noise_frac: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a dataset for predicting RC beam midspan deflection.

    Physics model (piecewise — slides §1-5):
        δ = P·L³ / (48·E·I_eff)

    Where I_eff switches at cracking:
        • Before cracking (M_max ≤ M_cr):  I_eff = I_gross
        • After cracking  (M_max  > M_cr):  I_eff = I_cr  (Branson formula simplified)

    Input features (6, all normalised to SI-like ranges):
        P       : Point load [kN]          ∈ [10, 500]
        L       : Beam length [m]          ∈ [2, 10]
        E       : Elastic modulus [GPa]    ∈ [25, 40]  (concrete)
        I_gross : Gross moment of inertia [×10⁻³ m⁴]  ∈ [1, 20]
        M_cr    : Cracking moment [kN·m]   ∈ [5, 80]
        rho     : Reinforcement ratio [-]  ∈ [0.005, 0.04]

    Output (1):
        delta : midspan deflection [mm]

    Parameters
    ----------
    n : int
        Number of samples
    noise_frac : float
        Gaussian noise fraction of output range
    random_state : int

    Returns
    -------
    X : np.ndarray, shape (n, 6)  — raw (unscaled) features
    y : np.ndarray, shape (n,)    — deflection in mm
    meta : dict                   — feature names, ranges, physics description
    """
    rng = np.random.default_rng(random_state)

    # Sample inputs from physical ranges
    P       = rng.uniform(10.0,    500.0,   n)      # kN
    L       = rng.uniform(2.0,     10.0,    n)      # m
    E       = rng.uniform(25.0,    40.0,    n) * 1e6  # kN/m² (GPa → kN/m²)
    I_gross = rng.uniform(1.0,     20.0,    n) * 1e-3  # m⁴
    M_cr    = rng.uniform(5.0,     80.0,    n)      # kN·m
    rho     = rng.uniform(0.005,   0.04,    n)      # dimensionless

    # Derived quantities
    M_max   = P * L / 4.0                            # kN·m (simply supported, midspan point load)

    # Branson's effective moment of inertia (simplified)
    #   I_cr ≈ I_gross * rho * (factor for tension steel contribution)
    I_cr    = I_gross * (rho * 10 + 0.1)            # crude approximation, bounded below gross
    I_cr    = np.minimum(I_cr, I_gross)

    # Piecewise stiffness
    I_eff   = np.where(M_max <= M_cr, I_gross,
                        I_cr + (I_gross - I_cr) * (M_cr / np.maximum(M_max, 1e-6)) ** 3)
    I_eff   = np.clip(I_eff, 0.01 * I_gross, I_gross)  # physical bounds

    # Midspan deflection: δ = PL³/(48EI)  [m → mm]
    delta   = (P * L ** 3) / (48.0 * E * I_eff) * 1000.0   # mm

    # Clip to physical range (deflection limits)
    delta   = np.clip(delta, 0.1, 200.0)

    # Add Gaussian noise
    delta_range = float(delta.max() - delta.min())
    noise = rng.normal(0, noise_frac * delta_range, n)
    delta = np.clip(delta + noise, 0.1, 220.0)

    # Build feature matrix (raw units → store, normalised separately in notebook)
    X = np.column_stack([
        P,
        L,
        E / 1e6,          # back to GPa for readability
        I_gross * 1e3,    # back to ×10⁻³ m⁴
        M_cr,
        rho,
    ])
    y = delta

    meta = {
        "feature_names": ["P (kN)", "L (m)", "E (GPa)", "I_gross (1e-3 m4)", "M_cr (kN·m)", "ρ (-)"],
        "target_name": "δ (mm)",
        "formula": "δ = P·L³/(48·E·I_eff), piecewise at M_max=M_cr",
        "n_samples": n,
        "noise_frac": noise_frac,
    }

    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
# CE Dataset 2 — Concrete Compressive Strength
# ─────────────────────────────────────────────────────────────────────────────

def generate_concrete_strength_dataset(
    n: int = 800,
    noise_std: float = 3.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a concrete compressive strength dataset.

    Uses a physics-inspired Abrams'-like model with interaction terms:
        f_c ≈ A / (w/c)^B × (1 + superplasticiser_factor) × curing_factor
            + slag_bonus + fly_ash_bonus + noise

    Input features (8):
        cement               [kg/m³]  ∈ [150, 540]
        water                [kg/m³]  ∈ [120, 230]
        fine_aggregate       [kg/m³]  ∈ [500, 950]
        coarse_aggregate     [kg/m³]  ∈ [700, 1150]
        fly_ash             [kg/m³]  ∈ [0, 200]
        slag                [kg/m³]  ∈ [0, 360]
        superplasticizer     [kg/m³]  ∈ [0, 32]
        curing_time          [days]   ∈ [3, 365]

    Output:
        f_c : 28-day compressive strength [MPa]

    References
    ----------
    Adapted from Yeh (1998) UCI Concrete Compressive Strength dataset.
    """
    rng = np.random.default_rng(random_state)

    cement      = rng.uniform(150,  540,  n)
    water       = rng.uniform(120,  230,  n)
    fine_agg    = rng.uniform(500,  950,  n)
    coarse_agg  = rng.uniform(700, 1150,  n)
    fly_ash     = rng.uniform(0,    200,  n)
    slag        = rng.uniform(0,    360,  n)
    sp          = rng.uniform(0,     32,  n)   # superplasticizer
    curing      = rng.uniform(3,    365,  n)   # days

    # Effective binder content
    total_binder = cement + 0.8 * fly_ash + 0.9 * slag

    # w/b ratio
    wb = water / np.maximum(total_binder, 1.0)

    # Abrams'-style base strength
    A, B = 115.0, 0.65
    f_base = A * (1 - wb) ** B

    # Superplasticizer effect (reduces effective w/b by up to 10%)
    sp_factor = 1.0 + 0.015 * np.log1p(sp)

    # Curing factor (logarithmic gain)
    curing_factor = 0.8 + 0.2 * np.log(curing / 28.0 + 1.0)

    f_c = f_base * sp_factor * curing_factor

    # Physical bounds [10, 90 MPa]
    f_c = np.clip(f_c, 10.0, 90.0)

    # Add noise
    f_c += rng.normal(0, noise_std, n)
    f_c = np.clip(f_c, 8.0, 95.0)

    X = np.column_stack([cement, water, fine_agg, coarse_agg, fly_ash, slag, sp, curing])
    y = f_c

    meta = {
        "feature_names": [
            "Cement (kg/m³)", "Water (kg/m³)", "Fine Agg (kg/m³)",
            "Coarse Agg (kg/m³)", "Fly Ash (kg/m³)", "Slag (kg/m³)",
            "Superplasticizer (kg/m³)", "Curing Time (days)"
        ],
        "target_name": "f_c (MPa)",
        "formula": "Abrams'-like: A*(1-w/b)^B * SP_factor * Curing_factor",
        "n_samples": n,
        "noise_std": noise_std,
    }

    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
# CE Dataset 3 — Traffic Flow Classification
# ─────────────────────────────────────────────────────────────────────────────

def generate_traffic_flow_dataset(
    n: int = 800,
    noise_std: float = 0.1,
    n_classes: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a tabular traffic flow classification dataset.

    Simulates traffic conditions at a highway sensor station.

    Input features (5):
        volume_upstream  : vehicles/hour  ∈ [0, 2400]
        speed_mean       : km/h           ∈ [10, 120]
        occupancy        : fraction       ∈ [0, 1]
        time_sin         : sin(time_of_day) ∈ [-1, 1]
        time_cos         : cos(time_of_day) ∈ [-1, 1]

    Output (n_classes):
        0 = Free flow      (speed > 80, occ < 0.2)
        1 = Congested      (40 < speed ≤ 80)
        2 = Heavy/stop-go  (speed ≤ 40)

    Parameters
    ----------
    n_classes : int
        2 (binary: free/not) or 3 (free/congested/heavy)
    """
    rng = np.random.default_rng(random_state)

    # Time of day (0-24h, cyclically encoded)
    hour = rng.uniform(0, 24, n)
    time_sin = np.sin(2 * np.pi * hour / 24)
    time_cos = np.cos(2 * np.pi * hour / 24)

    # Peak hour multiplier
    peak = 1.0 + 0.8 * np.exp(-((hour - 8) ** 2 / 4)) + 0.6 * np.exp(-((hour - 17.5) ** 2 / 3))

    # Speed (km/h) — inversely related to congestion
    speed_free = 110 + rng.normal(0, 5, n)
    congestion = np.clip(peak * rng.uniform(0.3, 1.0, n), 0.3, 1.5)
    speed_mean = np.clip(speed_free / congestion + rng.normal(0, noise_std * 20, n), 5, 125)

    # Volume (vehicles/hr)
    capacity = 2200.0
    volume_upstream = np.clip(capacity * congestion * 0.7 + rng.normal(0, 100, n), 0, 2400)

    # Occupancy
    occupancy = np.clip(congestion * 0.3 + rng.normal(0, noise_std * 0.1, n), 0, 1)

    X = np.column_stack([volume_upstream, speed_mean, occupancy, time_sin, time_cos])

    # Labels
    if n_classes == 2:
        y = (speed_mean <= 70).astype(int)
    else:
        y = np.zeros(n, dtype=int)
        y[speed_mean <= 70] = 1
        y[speed_mean <= 40] = 2

    meta = {
        "feature_names": ["Volume (veh/hr)", "Speed (km/h)", "Occupancy",
                           "Time (sin)", "Time (cos)"],
        "class_names": {2: ["Free", "Congested"],
                        3: ["Free", "Congested", "Heavy"]}[n_classes],
        "target_name": "Traffic Condition",
        "n_samples": n,
        "n_classes": n_classes,
    }

    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark / Toy Datasets
# ─────────────────────────────────────────────────────────────────────────────

def generate_xor_dataset(
    n: int = 400,
    noise: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    XOR classification dataset.

    Classic example showing linear model failure — used to motivate
    nonlinear neural networks.

    Returns
    -------
    X : (n, 2) — input features in [-1, 1]²
    y : (n,)   — binary labels {0, 1}
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)  # XOR rule
    X += rng.normal(0, noise, X.shape)
    return X, y


def generate_spiral_dataset(
    n: int = 300,
    noise: float = 0.1,
    n_classes: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spiral dataset — a challenging nonlinear classification benchmark.

    Points are arranged in n_classes interleaved spirals.
    Perfect for demonstrating why deep networks outperform shallow ones.

    Parameters
    ----------
    n : int
        Points per class
    n_classes : int
        Number of spiral arms (2 or 3)
    """
    rng = np.random.default_rng(random_state)
    X_list, y_list = [], []

    for c in range(n_classes):
        t = np.linspace(0, 1, n)
        angle = t * 3 * np.pi + (2 * np.pi * c / n_classes)
        r = 1 - t
        x1 = r * np.cos(angle) + rng.normal(0, noise, n)
        x2 = r * np.sin(angle) + rng.normal(0, noise, n)
        X_list.append(np.c_[x1, x2])
        y_list.append(np.full(n, c, dtype=int))

    return np.vstack(X_list), np.concatenate(y_list)


def generate_regression_1d(
    func_name: str = "sin",
    n: int = 200,
    noise: float = 0.1,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1-D regression dataset for demonstrating Universal Approximation.

    Parameters
    ----------
    func_name : str
        One of: 'sin', 'step', 'polynomial', 'abs', 'runge'
    n : int
        Number of points
    noise : float
        Gaussian noise std
    x_range : Tuple[float, float]

    Returns
    -------
    X : (n, 1)
    y : (n,)
    """
    rng = np.random.default_rng(random_state)
    x = np.linspace(x_range[0], x_range[1], n)

    functions = {
        "sin":        lambda x: np.sin(2 * np.pi * x),
        "step":       lambda x: (x > 0).astype(float),
        "polynomial": lambda x: 0.1 * x ** 3 - 0.5 * x ** 2 + x,
        "abs":        lambda x: np.abs(x),
        "runge":      lambda x: 1.0 / (1.0 + 25.0 * x ** 2),   # Runge's function
    }

    if func_name not in functions:
        raise ValueError(f"Unknown func_name '{func_name}'. Choose: {list(functions.keys())}")

    y = functions[func_name](x) + rng.normal(0, noise, n)
    return x.reshape(-1, 1), y


if __name__ == "__main__":
    X_beam, y_beam, meta = generate_beam_deflection_dataset(n=500)
    print(f"Beam dataset: X={X_beam.shape}, y={y_beam.shape}, range=[{y_beam.min():.1f}, {y_beam.max():.1f}] mm")

    X_conc, y_conc, meta = generate_concrete_strength_dataset(n=400)
    print(f"Concrete dataset: X={X_conc.shape}, y={y_conc.shape}, range=[{y_conc.min():.1f}, {y_conc.max():.1f}] MPa")

    X_traf, y_traf, meta = generate_traffic_flow_dataset(n=400)
    print(f"Traffic dataset: X={X_traf.shape}, y={y_traf.shape}, classes={np.unique(y_traf)}")

    X_xor, y_xor = generate_xor_dataset(n=400)
    print(f"XOR: X={X_xor.shape}, class balance={np.bincount(y_xor)}")

    X_sp, y_sp = generate_spiral_dataset(n=200)
    print(f"Spiral: X={X_sp.shape}")

    X_reg, y_reg = generate_regression_1d("runge", n=100)
    print(f"1D regression (runge): X={X_reg.shape}")

    print("ce_datasets.py loaded OK ✓")
