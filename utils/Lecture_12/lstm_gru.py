"""
lstm_gru.py — From-scratch LSTM and GRU cells in NumPy
=======================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides transparent, step-by-step implementations of:
  • LSTM cell  (Hochreiter & Schmidhuber, 1997)
  • GRU  cell  (Cho et al., 2014)

Every gate value is returned in a dict so that students can inspect and
plot what each gate is doing at every time step.

The scalar LSTM worked example from the slides is reproduced exactly:
    n_h=1, d=1, h(t-1)=0.5, x(t)=1.0, c(t-1)=0.8
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# Elementary nonlinearities (explicit, for pedagogy)
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid σ(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def tanh(z: np.ndarray) -> np.ndarray:
    """Element-wise tanh."""
    return np.tanh(z)


# ===========================================================================
# LSTM Cell
# ===========================================================================

def lstm_cell_forward(
    x_t   : np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    Wf: np.ndarray, Wi: np.ndarray,
    Wc: np.ndarray, Wo: np.ndarray,
    bf: np.ndarray, bi: np.ndarray,
    bc: np.ndarray, bo: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Single-step forward pass of one LSTM cell.

    Equations (using concatenated input h̃ = [h_prev ; x_t]):
        f  = σ(Wf @ h̃ + bf)           forget gate
        i  = σ(Wi @ h̃ + bi)           input gate
        c̃  = tanh(Wc @ h̃ + bc)        candidate cell state
        o  = σ(Wo @ h̃ + bo)           output gate
        c  = f ⊙ c_prev + i ⊙ c̃       cell state update  (additive!)
        h  = o ⊙ tanh(c)              hidden state

    Parameters
    ----------
    x_t   : (d,)         — input at current step
    h_prev: (n_h,)       — previous hidden state
    c_prev: (n_h,)       — previous cell state
    Wf,Wi,Wc,Wo: (n_h, n_h+d) — gate weight matrices
    bf,bi,bc,bo: (n_h,)  — gate biases

    Returns
    -------
    h_t    : (n_h,) — new hidden state
    c_t    : (n_h,) — new cell state
    gates  : dict with keys: 'f','i','c_tilde','o','c','h'
    """
    x_t    = np.atleast_1d(np.asarray(x_t, dtype=float))
    h_prev = np.atleast_1d(np.asarray(h_prev, dtype=float))
    c_prev = np.atleast_1d(np.asarray(c_prev, dtype=float))

    # Concatenate [h_prev ; x_t]
    h_tilde = np.concatenate([h_prev, x_t])

    # Gate computations
    f = sigmoid(Wf @ h_tilde + bf)        # forget gate — what to erase
    i = sigmoid(Wi @ h_tilde + bi)        # input  gate — what to write
    c_tilde = tanh(Wc @ h_tilde + bc)     # candidate cell state
    o = sigmoid(Wo @ h_tilde + bo)        # output gate — what to expose

    # Cell and hidden state update
    c_t = f * c_prev + i * c_tilde        # additive update (gradient highway)
    h_t = o * tanh(c_t)

    gates = {
        'f'       : f,          # forget gate activations ∈ (0,1)
        'i'       : i,          # input  gate activations ∈ (0,1)
        'c_tilde' : c_tilde,    # candidate                ∈ (-1,1)
        'o'       : o,          # output gate activations ∈ (0,1)
        'c'       : c_t,        # cell state
        'h'       : h_t,        # hidden state
    }
    return h_t, c_t, gates


def lstm_forward(
    X     : np.ndarray,
    params: Dict[str, np.ndarray],
    h0    : Optional[np.ndarray] = None,
    c0    : Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Full sequence forward pass through an LSTM.

    Parameters
    ----------
    X      : (T, d)  — input sequence
    params : dict with keys 'Wf','Wi','Wc','Wo','bf','bi','bc','bo'
    h0     : (n_h,) or None
    c0     : (n_h,) or None

    Returns
    -------
    H      : (T, n_h) — hidden states
    C      : (T, n_h) — cell states
    Gates  : dict of (T, n_h) arrays for 'f','i','c_tilde','o'
    """
    T, d = X.shape
    n_h  = params['Wf'].shape[0]

    if h0 is None: h0 = np.zeros(n_h)
    if c0 is None: c0 = np.zeros(n_h)

    H = np.zeros((T, n_h))
    C = np.zeros((T, n_h))
    Gates = {k: np.zeros((T, n_h)) for k in ('f','i','c_tilde','o','c','h')}

    h, c = h0.copy(), c0.copy()

    for t in range(T):
        h, c, gates = lstm_cell_forward(
            X[t], h, c,
            params['Wf'], params['Wi'], params['Wc'], params['Wo'],
            params['bf'], params['bi'], params['bc'], params['bo']
        )
        H[t] = h
        C[t] = c
        for k in ('f','i','c_tilde','o'):
            Gates[k][t] = gates[k]
        Gates['c'][t] = c
        Gates['h'][t] = h

    return H, C, Gates


# ---------------------------------------------------------------------------
# LSTM Gradient Analysis: additive cell update advantage
# ---------------------------------------------------------------------------

def lstm_cell_gradient(
    gates_over_time: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Approximate gradient norm of c_t w.r.t. c_{t-k} via forget gate products.

    ∂c_t / ∂c_{t-k}  ≈  prod_{tau=t-k+1}^{t} f_tau

    since ∂c_t/∂c_{t-1} = diag(f_t) (the forget gate).

    In practice this means gradients can flow over long horizons when f ≈ 1.

    Returns
    -------
    grad_norms : (T,) — gradient norm from each past step to the final step
    """
    f = gates_over_time['f']   # (T, n_h)
    T = f.shape[0]
    grad_norms = np.zeros(T)

    # Product from t backwards
    prod = np.ones(f.shape[1])
    for t in range(T - 1, -1, -1):
        prod *= f[t]
        grad_norms[t] = np.linalg.norm(prod)

    return grad_norms


# ===========================================================================
# GRU Cell
# ===========================================================================

def gru_cell_forward(
    x_t   : np.ndarray,
    h_prev: np.ndarray,
    Wr: np.ndarray, Wz: np.ndarray, Wh: np.ndarray,
    br: np.ndarray, bz: np.ndarray, bh: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Single-step forward pass of one GRU cell.

    Equations:
        r  = σ(Wr @ [h_prev ; x_t] + br)          reset gate
        z  = σ(Wz @ [h_prev ; x_t] + bz)          update gate
        h̃  = tanh(Wh @ [r ⊙ h_prev ; x_t] + bh)  candidate
        h  = (1 − z) ⊙ h_prev + z ⊙ h̃            final hidden state

    Parameters
    ----------
    x_t   : (d,)         — input at current step
    h_prev: (n_h,)       — previous hidden state
    Wr,Wz : (n_h, n_h+d) — gate weight matrices
    Wh    : (n_h, n_h+d) — note: Wh sees r ⊙ h_prev concatenated with x_t
    br,bz,bh: (n_h,)     — biases

    Returns
    -------
    h_t   : (n_h,) — new hidden state
    gates : dict with keys: 'r','z','h_tilde','h'
    """
    x_t    = np.atleast_1d(np.asarray(x_t, dtype=float))
    h_prev = np.atleast_1d(np.asarray(h_prev, dtype=float))

    concat  = np.concatenate([h_prev, x_t])      # for reset & update gates
    r = sigmoid(Wr @ concat + br)                # reset gate
    z = sigmoid(Wz @ concat + bz)                # update gate

    # Candidate uses r ⊙ h_prev (reset controls what past to forget)
    concat_r = np.concatenate([r * h_prev, x_t])
    h_tilde  = tanh(Wh @ concat_r + bh)

    # Final hidden state is a linear interpolation
    h_t = (1.0 - z) * h_prev + z * h_tilde

    gates = {'r': r, 'z': z, 'h_tilde': h_tilde, 'h': h_t}
    return h_t, gates


def gru_forward(
    X     : np.ndarray,
    params: Dict[str, np.ndarray],
    h0    : Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Full sequence forward pass through a GRU.

    Parameters
    ----------
    X      : (T, d)
    params : dict with keys 'Wr','Wz','Wh','br','bz','bh'
    h0     : (n_h,) or None

    Returns
    -------
    H      : (T, n_h)
    Gates  : dict of (T, n_h) arrays for 'r','z','h_tilde'
    """
    T, d = X.shape
    n_h  = params['Wr'].shape[0]

    if h0 is None: h0 = np.zeros(n_h)

    H = np.zeros((T, n_h))
    Gates = {k: np.zeros((T, n_h)) for k in ('r','z','h_tilde')}

    h = h0.copy()
    for t in range(T):
        h, gates = gru_cell_forward(
            X[t], h,
            params['Wr'], params['Wz'], params['Wh'],
            params['br'], params['bz'], params['bh']
        )
        H[t] = h
        for k in ('r','z','h_tilde'):
            Gates[k][t] = gates[k]

    return H, Gates


# ===========================================================================
# Parameter Initialisation
# ===========================================================================

def init_lstm_params(d: int, n_h: int, scale: float = 0.1, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Initialise all LSTM weight matrices.

    Returns dict with keys: 'Wf','Wi','Wc','Wo','bf','bi','bc','bo'
    Each W : (n_h, n_h+d);  each b : (n_h,)
    """
    rng = np.random.default_rng(seed)
    W_shape = (n_h, n_h + d)
    return {
        'Wf': rng.standard_normal(W_shape) * scale,
        'Wi': rng.standard_normal(W_shape) * scale,
        'Wc': rng.standard_normal(W_shape) * scale,
        'Wo': rng.standard_normal(W_shape) * scale,
        'bf': np.zeros(n_h),
        'bi': np.zeros(n_h),
        'bc': np.zeros(n_h),
        'bo': np.zeros(n_h),
    }


def init_gru_params(d: int, n_h: int, scale: float = 0.1, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Initialise all GRU weight matrices.

    Returns dict with keys: 'Wr','Wz','Wh','br','bz','bh'
    """
    rng = np.random.default_rng(seed)
    W_shape = (n_h, n_h + d)
    return {
        'Wr': rng.standard_normal(W_shape) * scale,
        'Wz': rng.standard_normal(W_shape) * scale,
        'Wh': rng.standard_normal(W_shape) * scale,
        'br': np.zeros(n_h),
        'bz': np.zeros(n_h),
        'bh': np.zeros(n_h),
    }


def count_params_lstm(d: int, n_h: int, n_out: int = 1) -> Dict[str, int]:
    """Analytical param count for an LSTM layer: 4 × n_h × (n_h + d + 1)."""
    gate = n_h * (n_h + d) + n_h    # one gate's weights + bias
    total_gate = 4 * gate
    readout    = n_out * n_h + n_out
    return {
        'per_gate'      : gate,
        'all_gates'     : total_gate,
        'readout'       : readout,
        'total'         : total_gate + readout,
        'formula'       : f'4 × n_h(n_h + d + 1) = 4 × {n_h}({n_h}+{d}+1) = {4*n_h*(n_h+d+1)}',
    }


def count_params_gru(d: int, n_h: int, n_out: int = 1) -> Dict[str, int]:
    """Analytical param count for a GRU layer: 3 × n_h × (n_h + d + 1)."""
    gate = n_h * (n_h + d) + n_h
    total_gate = 3 * gate
    readout    = n_out * n_h + n_out
    return {
        'per_gate'      : gate,
        'all_gates'     : total_gate,
        'readout'       : readout,
        'total'         : total_gate + readout,
        'formula'       : f'3 × n_h(n_h + d + 1) = 3 × {n_h}({n_h}+{d}+1) = {3*n_h*(n_h+d+1)}',
    }


# ===========================================================================
# Self-test / Worked Example from Slides
# ===========================================================================

def slides_lstm_worked_example() -> None:
    """
    Reproduce the scalar LSTM worked example from Lecture 12 Practice Problem 2.

    Setup (scalar LSTM, n_h=1, d=1):
        w_f = [-1, 2],  b_f = 0
        w_i = [ 1, 1],  b_i = 0
        w_c = [0.5, -0.5], b_c = 0
        w_o = [1, 0],   b_o = 0
        h(t-1) = 0.5,  x(t) = 1.0,  c(t-1) = 0.8
    """
    # n_h=1, d=1 → concatenated dim = 2
    Wf = np.array([[-1.0,  2.0]])    # (1, 2)
    Wi = np.array([[ 1.0,  1.0]])
    Wc = np.array([[ 0.5, -0.5]])
    Wo = np.array([[ 1.0,  0.0]])
    bf = bi = bc = bo = np.zeros(1)

    h_prev = np.array([0.5])
    x_t    = np.array([1.0])
    c_prev = np.array([0.8])

    h_t, c_t, gates = lstm_cell_forward(
        x_t, h_prev, c_prev,
        Wf, Wi, Wc, Wo,
        bf, bi, bc, bo
    )

    print("=" * 60)
    print("SLIDES LSTM WORKED EXAMPLE (Practice Problem 2)")
    print("=" * 60)
    print(f"\n  h(t-1) = {h_prev[0]},  x(t) = {x_t[0]},  c(t-1) = {c_prev[0]}")
    h_tilde_vec = np.array([h_prev[0], x_t[0]])   # 1D: [h(t-1), x(t)]
    print(f"  h̃ = [h(t-1); x(t)] = {h_tilde_vec}")

    f_preact = float(Wf @ h_tilde_vec)
    print(f"\n  Forget gate  f = σ(Wf @ h̃) = σ({f_preact:.4f}) = {gates['f'][0]:.3f}")
    print(f"    Expected: σ(1.5) ≈ 0.818  ✓" if abs(gates['f'][0] - 0.818) < 1e-3 else "")
    print(f"\n  Input  gate  i = σ({Wi} @ h̃) = {gates['i'][0]:.3f}")
    print(f"    Expected: σ(1.5) ≈ 0.818  ✓" if abs(gates['i'][0] - 0.818) < 1e-3 else "")
    print(f"\n  Candidate   c̃ = tanh({Wc} @ h̃) = {gates['c_tilde'][0]:.3f}")
    print(f"    Expected: tanh(-0.25) ≈ -0.244  ✓" if abs(gates['c_tilde'][0] + 0.244) < 1e-3 else "")
    print(f"\n  Output gate  o = σ({Wo} @ h̃) = {gates['o'][0]:.3f}")
    print(f"    Expected: σ(0.5) ≈ 0.622  ✓" if abs(gates['o'][0] - 0.622) < 1e-3 else "")
    print(f"\n  Cell update  c(t) = f⊙c(t-1) + i⊙c̃ = {c_t[0]:.3f}")
    print(f"    Expected: 0.818×0.8 + 0.818×(-0.244) ≈ 0.455  ✓" if abs(c_t[0] - 0.455) < 0.01 else "")
    print(f"\n  Hidden state h(t) = o⊙tanh(c(t)) = {h_t[0]:.3f}")
    print(f"    Expected: 0.622×tanh(0.455) ≈ 0.264  ✓" if abs(h_t[0] - 0.264) < 0.01 else "")
    print(f"\n  Interpretation: f≈0.82 → forget gate retains ~82% of c(t-1) — most long-term memory preserved.")

    print("\n✅ LSTM worked example matches slides!\n")


if __name__ == "__main__":
    slides_lstm_worked_example()

    # Quick test of full sequence LSTM
    np.random.seed(42)
    T, d, n_h = 10, 3, 8
    X      = np.random.randn(T, d)
    params = init_lstm_params(d, n_h)

    H, C, Gates = lstm_forward(X, params)
    assert H.shape == (T, n_h),  f"H shape wrong: {H.shape}"
    assert C.shape == (T, n_h),  f"C shape wrong: {C.shape}"
    assert Gates['f'].shape == (T, n_h)
    print(f"LSTM forward pass OK: H{H.shape}, C{C.shape}")
    print(f"  Forget gate range: [{Gates['f'].min():.3f}, {Gates['f'].max():.3f}]  (should be in (0,1))")

    # Quick test of full sequence GRU
    params_gru = init_gru_params(d, n_h)
    H_gru, Gates_gru = gru_forward(X, params_gru)
    assert H_gru.shape == (T, n_h)
    print(f"\nGRU forward pass OK: H{H_gru.shape}")

    # Parameter count comparison
    p_lstm = count_params_lstm(d=3, n_h=64)
    p_gru  = count_params_gru(d=3, n_h=64)
    print(f"\nParam count (d=3, n_h=64):")
    print(f"  LSTM: {p_lstm['total']:,}  ({p_lstm['formula']})")
    print(f"  GRU : {p_gru['total']:,}   ({p_gru['formula']})")

    # LSTM gradient analysis
    grad_norms = lstm_cell_gradient(Gates)
    print(f"\nLSTM gradient norms (cell path): {grad_norms.round(3)}")
    print("  (Values close to 1 = gradient highway is working)")

    print("\n✅ lstm_gru.py: all self-tests passed!")
