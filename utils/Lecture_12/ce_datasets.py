"""
ce_datasets.py — Synthetic CE Time-Series Generators
=====================================================
CE 639: AI for Civil Engineering — Lecture 12

Generates realistic synthetic time-series data for all CE applications
covered in the lecture, with controllable noise, seasonality, and events.

All generators return NumPy arrays ready to pass into training.create_sequences().

Datasets:
  1. Streamflow Forecasting  (hydrology)
  2. Structural Health Monitoring (vibration signals)
  3. Traffic Flow Forecasting
  4. Air Quality (PM2.5) Forecasting
  5. Construction Progress Monitoring
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter1d


# ===========================================================================
# 1. Streamflow Forecasting
# ===========================================================================

def generate_streamflow(
    n_days       : int   = 730,   # 2 years
    n_features   : int   = 3,     # [rainfall, temperature, soil_moisture]
    seasonality  : bool  = True,
    add_events   : bool  = True,  # flash flood / drought events
    noise_level  : float = 0.15,
    random_state : int   = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic daily streamflow data (hydrometeorological inputs → Q).

    The generating equations loosely follow a nonlinear conceptual model:
        Q_t = f(P_t, T_t, θ_t)  with memory from previous days.

    Features:
        Col 0: Rainfall P_t  [mm/day]
        Col 1: Temperature T_t [°C]
        Col 2: Antecedent soil moisture θ_t [0–1]
    Target:
        Streamflow Q_t [m³/s]

    Parameters
    ----------
    n_days       : length of the time series
    n_features   : must be ≤ 3 (always generates all 3, but returns first n)
    seasonality  : add annual cycle to temperature
    add_events   : insert 2–4 flash-flood peaks and a drought period
    noise_level  : relative noise amplitude
    random_state : RNG seed

    Returns
    -------
    data     : (n_days, n_features+1) — last column is streamflow Q
    times    : (n_days,)  day indices 0..n_days-1
    metadata : dict with column names and event info
    """
    rng  = np.random.default_rng(random_state)
    t    = np.arange(n_days, dtype=float)

    # ---- Rainfall (mm/day) ----
    # Base: occasional rain events
    rain_prob = 0.25 + 0.10 * np.sin(2 * np.pi * t / 365)   # seasonal
    rain_mask = rng.random(n_days) < rain_prob
    rainfall  = np.where(rain_mask,
                         rng.exponential(scale=8.0, size=n_days),
                         0.0)
    # Flash flood events
    if add_events:
        event_days = rng.integers(30, n_days - 30, size=3)
        for ed in event_days:
            rainfall[ed:ed+3] += rng.uniform(40, 80, size=3)

    # ---- Temperature (°C) ----
    if seasonality:
        temperature = 20 + 10 * np.sin(2 * np.pi * t / 365 - np.pi / 2) \
                      + rng.normal(0, 2, n_days)
    else:
        temperature = np.full(n_days, 20.0) + rng.normal(0, 2, n_days)

    # ---- Soil Moisture (0–1) ----
    # Simple bucket model: θ_{t+1} = θ_t + P_t/150 - ET_t
    soil_moisture = np.zeros(n_days)
    soil_moisture[0] = 0.4
    et_rate = 0.003 + 0.002 * np.sin(2 * np.pi * t / 365)   # evapotranspiration
    for i in range(1, n_days):
        soil_moisture[i] = np.clip(
            soil_moisture[i-1] + rainfall[i] / 150.0 - et_rate[i], 0.05, 0.95
        )

    # ---- Streamflow (m³/s) — nonlinear model ----
    # Q depends on current rain, antecedent soil moisture, and memory
    Q = np.zeros(n_days)
    Q[0] = 10.0
    decay = 0.85  # baseflow recession coefficient
    for i in range(1, n_days):
        # Nonlinear rainfall-runoff
        runoff_coeff = soil_moisture[i] ** 1.5     # more runoff when wet
        runoff       = rainfall[i] * runoff_coeff * rng.uniform(0.8, 1.2)
        # Baseflow recession
        Q[i] = max(decay * Q[i-1] + runoff + 2.0, 0.5)

    # Drought: suppress Q for a stretch
    if add_events and n_days > 200:
        drought_start = rng.integers(60, n_days // 2)
        drought_len   = rng.integers(20, min(45, n_days // 4))
        Q[drought_start : drought_start + drought_len] *= 0.3

    # Add noise
    Q *= (1 + rng.normal(0, noise_level, n_days))
    Q  = np.clip(Q, 0.1, None)

    # Smooth slightly (routing delay)
    Q = gaussian_filter1d(Q, sigma=1.5)

    # Assemble dataset
    features = np.stack([rainfall, temperature, soil_moisture], axis=1)
    data     = np.concatenate([features[:, :n_features], Q[:, None]], axis=1)

    metadata = {
        'columns'      : ['Rainfall (mm)', 'Temperature (°C)', 'Soil Moisture'][:n_features] + ['Streamflow (m³/s)'],
        'target_col'   : n_features,    # last column is Q
        'n_days'       : n_days,
        'lookback_hint': 30,            # domain: 30-day antecedent window
    }
    return data.astype(np.float32), t.astype(np.float32), metadata


# ===========================================================================
# 2. Structural Health Monitoring (SHM)
# ===========================================================================

def generate_vibration_signals(
    n_signals   : int   = 200,
    length      : int   = 500,    # samples per signal
    fs          : float = 200.0,  # sampling frequency [Hz]
    damage_level: float = 0.0,    # 0 = healthy, 1 = severely damaged
    add_noise   : bool  = True,
    random_state: int   = 42
) -> np.ndarray:
    """
    Generate synthetic SDOF free-vibration acceleration signals.

    The SDOF equation is:  m·ü + c·u̇ + k·u = F(t)

    Damage modifies stiffness k (frequency shift) and damping c (amplitude).

    Parameters
    ----------
    n_signals    : number of signals to generate
    length       : samples per signal
    fs           : sampling frequency [Hz]
    damage_level : 0 (healthy) → 1 (heavily damaged)
    add_noise    : add sensor measurement noise
    random_state : RNG seed

    Returns
    -------
    signals : (n_signals, length)  acceleration time series [g]
    """
    rng = np.random.default_rng(random_state)
    t   = np.linspace(0, length / fs, length)
    signals = np.zeros((n_signals, length))

    for i in range(n_signals):
        # Natural frequency (Hz): damage shifts it down
        f_n    = rng.uniform(4.0, 6.0)
        f_n_d  = f_n * (1.0 - damage_level * 0.25)   # damaged freq

        # Damping ratio: increases with damage
        zeta   = 0.02 + damage_level * 0.06 + rng.uniform(-0.005, 0.005)
        omega_n = 2 * np.pi * f_n_d
        omega_d = omega_n * np.sqrt(max(1 - zeta**2, 1e-4))  # damped freq

        # Free vibration with impulse initial condition
        A0  = rng.uniform(0.5, 2.0)          # initial amplitude [g]
        phi = rng.uniform(0, 2 * np.pi)       # phase

        sig = A0 * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t + phi)

        # Damage adds harmonic distortion (nonlinear response)
        if damage_level > 0:
            sig += damage_level * 0.3 * A0 * np.exp(-zeta * omega_n * t) \
                   * np.sin(2 * omega_d * t + phi)

        # Measurement noise
        if add_noise:
            sig += rng.normal(0, 0.02 + damage_level * 0.05, length)

        signals[i] = sig

    return signals.astype(np.float32)


def generate_shm_dataset(
    n_per_class  : int   = 100,
    signal_length: int   = 500,
    fs           : float = 200.0,
    damage_levels: Tuple = (0.0, 0.3, 0.7, 1.0),
    random_state : int   = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Labelled SHM dataset: signals × damage states.

    Returns
    -------
    X        : (N, signal_length, 1) — signals as univariate sequences
    y        : (N,)                  — damage class labels (0=healthy...)
    metadata : dict
    """
    all_X, all_y = [], []
    for label, dl in enumerate(damage_levels):
        sigs = generate_vibration_signals(
            n_signals=n_per_class, length=signal_length,
            fs=fs, damage_level=dl,
            random_state=random_state + label
        )
        all_X.append(sigs[:, :, None])   # add feature dim
        all_y.append(np.full(n_per_class, label, dtype=np.int64))

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0)

    metadata = {
        'classes'     : [f'Damage {int(dl*100)}%' for dl in damage_levels],
        'n_classes'   : len(damage_levels),
        'fs'          : fs,
        'signal_length': signal_length,
    }
    return X, y, metadata


# ===========================================================================
# 3. Traffic Flow Forecasting
# ===========================================================================

def generate_traffic_data(
    n_days       : int   = 365,
    interval_min : int   = 15,       # observation interval in minutes
    n_routes     : int   = 1,
    add_accidents: bool  = True,
    random_state : int   = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic traffic flow time series.

    Models:
      • Diurnal cycle (morning/evening rush hours)
      • Weekly seasonality (weekday vs weekend)
      • Random disruptions / accidents

    Features per time step:
        Col 0: Traffic volume [vehicles/interval]
        Col 1: Speed            [km/h]
        Col 2: Occupancy       [0–1]

    Returns
    -------
    data     : (N, 3)  where N = n_days × (1440 // interval_min)
    times    : (N,)    minutes from start
    metadata : dict
    """
    rng           = np.random.default_rng(random_state)
    steps_per_day = 1440 // interval_min
    N             = n_days * steps_per_day
    t_min         = np.arange(N) * interval_min     # absolute minute index
    hour_of_day   = (t_min % 1440) / 60             # 0–24

    # ---- Diurnal volume pattern ----
    # Morning peak ~8am, evening peak ~6pm
    morning_rush = np.exp(-0.5 * ((hour_of_day - 8.0) / 1.2) ** 2)
    evening_rush = np.exp(-0.5 * ((hour_of_day - 18.0) / 1.5) ** 2)
    base_volume  = 100 + 800 * morning_rush + 900 * evening_rush

    # Weekend factor
    day_of_week      = (t_min // 1440) % 7
    weekend_mask     = (day_of_week >= 5)
    base_volume[weekend_mask] *= 0.6

    # Random variation
    volume = base_volume * rng.uniform(0.85, 1.15, N)

    # Accidents / disruptions
    if add_accidents:
        n_events = rng.integers(5, 15)
        for _ in range(n_events):
            start = rng.integers(0, N - 60)
            dur   = rng.integers(15, 90) // interval_min
            volume[start : start + dur] *= rng.uniform(0.2, 0.6)

    volume = np.clip(volume, 0, None)

    # ---- Speed (inversely related to volume) ----
    max_speed = 90.0   # km/h
    jam_density = 1200.0
    speed = max_speed * np.maximum(1 - volume / jam_density, 0.05)
    speed += rng.normal(0, 3, N)
    speed = np.clip(speed, 5, max_speed)

    # ---- Occupancy (density proxy) ----
    occupancy = volume / jam_density + rng.normal(0, 0.02, N)
    occupancy = np.clip(occupancy, 0, 1)

    data = np.stack([volume, speed, occupancy], axis=1).astype(np.float32)

    metadata = {
        'columns'     : ['Volume (veh/interval)', 'Speed (km/h)', 'Occupancy'],
        'target_col'  : 0,
        'interval_min': interval_min,
        'steps_per_day': steps_per_day,
        'lookback_hint': steps_per_day * 2,   # 2 days context
    }
    return data, t_min.astype(np.float32), metadata


# ===========================================================================
# 4. Air Quality Forecasting (PM2.5)
# ===========================================================================

def generate_air_quality(
    n_days      : int   = 365,
    random_state: int   = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic daily air quality time series.

    Inspired by advection-diffusion dynamics: PM2.5 concentrations depend
    on emissions, wind speed, humidity, and antecedent concentrations.

    Features (6 columns):
        0: PM2.5   [μg/m³]
        1: NO2     [ppb]
        2: O3      [ppb]
        3: Temperature [°C]
        4: Wind Speed  [m/s]
        5: Humidity    [%]
    Target: PM2.5 next day (col 0, shifted by 1)

    Returns
    -------
    data     : (n_days, 6)
    times    : (n_days,)
    metadata : dict
    """
    rng  = np.random.default_rng(random_state)
    t    = np.arange(n_days, dtype=float)

    # ---- Meteorological variables ----
    temperature = 25 + 8 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, n_days)
    wind_speed  = np.abs(3 + 2 * np.sin(2 * np.pi * t / 365 + 1) + rng.normal(0, 1, n_days))
    humidity    = 60 + 20 * np.cos(2 * np.pi * t / 365) + rng.normal(0, 5, n_days)
    humidity    = np.clip(humidity, 20, 95)

    # ---- PM2.5 (advection-diffusion inspired) ----
    # Higher in winter, higher with low wind, higher with high humidity
    pm25 = np.zeros(n_days)
    pm25[0] = 35.0
    emission = 30 + 10 * np.sin(2 * np.pi * t / 365 + np.pi)   # winter peak

    for i in range(1, n_days):
        dispersion  = wind_speed[i] * 3.0
        accumulation = humidity[i] * 0.2
        pm25[i] = (0.7 * pm25[i-1]
                   + emission[i]
                   - dispersion
                   + accumulation
                   + rng.normal(0, 5))
        pm25[i] = max(pm25[i], 1.0)

    # ---- NO2 and O3 ----
    no2 = 20 + 10 * np.sin(2 * np.pi * t / 365) + 0.3 * pm25 + rng.normal(0, 3, n_days)
    o3  = 40 - 0.2 * no2 + 15 * np.sin(2 * np.pi * t / 365 - 1) + rng.normal(0, 4, n_days)
    no2 = np.clip(no2, 0, None)
    o3  = np.clip(o3, 0, None)

    data = np.stack([pm25, no2, o3, temperature, wind_speed, humidity], axis=1)

    metadata = {
        'columns'     : ['PM2.5 (μg/m³)', 'NO2 (ppb)', 'O3 (ppb)',
                         'Temperature (°C)', 'Wind Speed (m/s)', 'Humidity (%)'],
        'target_col'  : 0,     # predict PM2.5
        'lookback_hint': 7,    # 1-week context
        'who_limit_pm25': 15,  # WHO 24h guideline μg/m³
    }
    return data.astype(np.float32), t.astype(np.float32), metadata


# ===========================================================================
# 5. Construction Progress Monitoring
# ===========================================================================

def generate_construction_progress(
    n_weeks    : int   = 52,   # 1-year project
    n_projects : int   = 50,
    delay_prob : float = 0.4,  # probability of a project experiencing delay
    random_state: int  = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic weekly construction progress data.

    Features per week:
        0: Completion %     [0–100]
        1: Resource util.   [0–1]
        2: Weather stops    [days/week]
        3: Milestone achieved [0/1]
        4: Planned %        [0–100]

    Target:
        1 if the project is delayed at week 4 in the future, else 0.

    Note: This is a path-dependent task — the trajectory of progress,
    not just the current completion %, predicts delay.

    Returns
    -------
    X        : (n_projects, n_weeks, 5)
    y        : (n_projects,)  0=on-time, 1=delayed
    metadata : dict
    """
    rng = np.random.default_rng(random_state)

    X = np.zeros((n_projects, n_weeks, 5), dtype=np.float32)
    y = np.zeros(n_projects,               dtype=np.int64)

    for p in range(n_projects):
        delayed   = rng.random() < delay_prob
        y[p]      = 1 if delayed else 0

        # Planned linear progress
        planned = np.linspace(0, 100, n_weeks)

        # Actual progress
        actual = np.zeros(n_weeks)
        actual[0] = rng.uniform(0, 3)

        cumulative_delay = 0
        for w in range(1, n_weeks):
            # Weekly increment
            nominal_increment = 100 / n_weeks * rng.uniform(0.7, 1.3)

            # Delay events
            if delayed and rng.random() < 0.2:
                nominal_increment *= rng.uniform(0.1, 0.5)
                cumulative_delay  += 1

            actual[w] = min(actual[w-1] + nominal_increment, 100)

        # Resource utilisation (0–1)
        resource = np.clip(0.7 + rng.normal(0, 0.1, n_weeks), 0.1, 1.0)
        if delayed:
            resource[rng.integers(0, n_weeks//2)] *= 0.4  # resource disruption

        # Weather stoppages (days/week)
        weather = np.clip(rng.poisson(0.5, n_weeks) * (1 + 0.5 * delayed), 0, 5)

        # Milestone flags (at 25%, 50%, 75%, 100%)
        milestones = ((actual >= 25) & np.roll(actual < 25, 1)).astype(float)
        milestones += ((actual >= 50) & np.roll(actual < 50, 1)).astype(float)
        milestones += ((actual >= 75) & np.roll(actual < 75, 1)).astype(float)

        X[p, :, 0] = actual
        X[p, :, 1] = resource
        X[p, :, 2] = weather
        X[p, :, 3] = milestones
        X[p, :, 4] = planned

    metadata = {
        'columns'  : ['Completion %', 'Resource Util', 'Weather Stops',
                      'Milestone', 'Planned %'],
        'target'   : 'Delayed (binary)',
        'n_classes': 2,
        'prediction_horizon_weeks': 4,
    }
    return X, y, metadata


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    print("Testing ce_datasets.py...")

    # 1. Streamflow
    data, t, meta = generate_streamflow(n_days=365)
    print(f"  Streamflow: data{data.shape}, cols={meta['columns']}")
    assert data.shape == (365, 4), f"Streamflow shape: {data.shape}"

    # 2. SHM signals
    sigs = generate_vibration_signals(n_signals=20, length=500)
    print(f"  Vibration signals: {sigs.shape}")
    assert sigs.shape == (20, 500)

    X_shm, y_shm, meta_shm = generate_shm_dataset(n_per_class=25)
    print(f"  SHM dataset: X{X_shm.shape}, y{y_shm.shape}, classes={meta_shm['classes']}")

    # 3. Traffic
    data_t, t_min, meta_t = generate_traffic_data(n_days=7, interval_min=15)
    print(f"  Traffic: data{data_t.shape}")
    assert data_t.shape == (7 * 96, 3), f"Traffic shape: {data_t.shape}"

    # 4. Air Quality
    data_aq, t_aq, meta_aq = generate_air_quality(n_days=365)
    print(f"  Air quality: data{data_aq.shape}, cols={meta_aq['columns']}")

    # 5. Construction
    X_c, y_c, meta_c = generate_construction_progress(n_weeks=52, n_projects=50)
    print(f"  Construction: X{X_c.shape}, y{y_c.shape}")
    print(f"  Delay rate: {y_c.mean():.2f}  (expected ≈ 0.40)")
    assert X_c.shape == (50, 52, 5)

    print("\n✅ ce_datasets.py: all self-tests passed!")
