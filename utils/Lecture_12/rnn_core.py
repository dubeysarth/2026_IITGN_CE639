"""
rnn_core.py — From-scratch Vanilla (Elman) RNN in NumPy
=========================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides a step-by-step, fully transparent implementation of the Elman RNN
using only NumPy.  Every function includes shape assertions and docstrings
so that students can trace exactly what happens at each time step.

The worked example from the slides is reproduced exactly:
    d=1, n_h=2, tanh activation, 2-step forward pass.
"""

import numpy as np
from typing import Optional, Tuple, Generator, Dict, Any


# ---------------------------------------------------------------------------
# Core RNN Operations
# ---------------------------------------------------------------------------

def tanh_activation(z: np.ndarray) -> np.ndarray:
    """Element-wise tanh, made explicit so students see the nonlinearity."""
    return np.tanh(z)


def rnn_cell_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    bh: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-step forward pass of one Elman RNN cell.

    Computes:
        z_t  = Wh @ h_prev + Wx @ x_t + bh
        h_t  = tanh(z_t)

    Parameters
    ----------
    x_t   : (d,)       — input at current time step
    h_prev: (n_h,)     — hidden state from previous step
    Wx    : (n_h, d)   — input-to-hidden weights
    Wh    : (n_h, n_h) — hidden-to-hidden (recurrent) weights
    bh    : (n_h,)     — hidden bias

    Returns
    -------
    h_t   : (n_h,) — updated hidden state
    z_t   : (n_h,) — pre-activation (for BPTT and gradient demos)
    """
    x_t    = np.atleast_1d(np.asarray(x_t, dtype=float))
    h_prev = np.atleast_1d(np.asarray(h_prev, dtype=float))

    assert Wx.shape == (Wh.shape[0], x_t.shape[0]),  \
        f"Wx shape mismatch: expected {(Wh.shape[0], x_t.shape[0])}, got {Wx.shape}"
    assert Wh.shape[0] == Wh.shape[1],                \
        f"Wh must be square (n_h × n_h), got {Wh.shape}"
    assert h_prev.shape[0] == Wh.shape[0],            \
        f"h_prev dim mismatch: expected {Wh.shape[0]}, got {h_prev.shape[0]}"

    z_t = Wh @ h_prev + Wx @ x_t + bh          # pre-activation
    h_t = tanh_activation(z_t)                   # hidden state update
    return h_t, z_t


def rnn_forward(
    X: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    bh: np.ndarray,
    h0: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full forward pass of an Elman RNN over a sequence.

    Parameters
    ----------
    X   : (T, d)   — input sequence (T time steps, d features)
    Wx  : (n_h, d)
    Wh  : (n_h, n_h)
    bh  : (n_h,)
    h0  : (n_h,) or None — initial hidden state (zeros if None)

    Returns
    -------
    H   : (T, n_h) — hidden states at every time step
    Z   : (T, n_h) — pre-activations (useful for gradient analysis)
    """
    T, d = X.shape
    n_h  = Wh.shape[0]

    if h0 is None:
        h0 = np.zeros(n_h)

    H = np.zeros((T, n_h))
    Z = np.zeros((T, n_h))
    h = h0.copy()

    for t in range(T):
        h, z   = rnn_cell_forward(X[t], h, Wx, Wh, bh)
        H[t]   = h
        Z[t]   = z

    return H, Z


def rnn_forward_step_by_step(
    X: np.ndarray,
    Wx: np.ndarray,
    Wh: np.ndarray,
    bh: np.ndarray,
    h0: Optional[np.ndarray] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Generator version of rnn_forward — yields one dict per time step.

    Useful for animation (animate_rnn_forward) and interactive inspection.

    Yields (per time step t)
    -------------------------
    {
      't'     : int,         — time step index (0-based)
      'x_t'   : (d,),        — current input
      'h_prev': (n_h,),      — previous hidden state
      'z_t'   : (n_h,),      — pre-activation
      'h_t'   : (n_h,),      — updated hidden state
      'H_so_far': (t+1, n_h) — all hidden states computed so far
    }
    """
    T, d = X.shape
    n_h  = Wh.shape[0]

    if h0 is None:
        h0 = np.zeros(n_h)

    H_so_far = np.zeros((T, n_h))
    h = h0.copy()

    for t in range(T):
        h_prev = h.copy()
        h, z   = rnn_cell_forward(X[t], h, Wx, Wh, bh)
        H_so_far[t] = h

        yield {
            't'         : t,
            'x_t'       : X[t],
            'h_prev'    : h_prev,
            'z_t'       : z,
            'h_t'       : h,
            'H_so_far'  : H_so_far[:t+1].copy()
        }


def rnn_predict(
    H: np.ndarray,
    Wy: np.ndarray,
    by: np.ndarray,
    output_at: str = 'last'
) -> np.ndarray:
    """
    Linear read-out layer applied to hidden states.

    Parameters
    ----------
    H        : (T, n_h)             — hidden states from rnn_forward
    Wy       : (n_out, n_h)         — output weights
    by       : (n_out,)             — output bias
    output_at: 'last' | 'all'       — predict at final step or all steps

    Returns
    -------
    y_hat : (n_out,) if 'last', or (T, n_out) if 'all'
    """
    if output_at == 'last':
        return Wy @ H[-1] + by
    elif output_at == 'all':
        return H @ Wy.T + by
    else:
        raise ValueError(f"output_at must be 'last' or 'all', got '{output_at}'")


# ---------------------------------------------------------------------------
# BPTT Gradient Analysis
# ---------------------------------------------------------------------------

def compute_bptt_gradient_norms(
    Z: np.ndarray,
    Wh: np.ndarray
) -> np.ndarray:
    """
    Compute the approximate gradient norm flowing backwards through time.

    For an RNN of length T, the gradient at step k wrt loss at step T involves:
        prod_{tau=k+1}^{T} diag(tanh'(z_tau)) @ Wh

    We track the spectral norm of this accumulated product.

    Parameters
    ----------
    Z   : (T, n_h) — pre-activations from rnn_forward
    Wh  : (n_h, n_h) — recurrent weight matrix

    Returns
    -------
    grad_norms : (T,) — gradient norm from each time step back to step 0
    """
    T, n_h = Z.shape

    # tanh'(z) = 1 - tanh(z)^2
    dtanh = 1.0 - np.tanh(Z) ** 2   # (T, n_h)

    # Start from the end and accumulate: J_t = diag(dtanh[T-1]) @ Wh
    J_accum = np.eye(n_h)
    grad_norms = np.zeros(T)

    for t in range(T - 1, -1, -1):
        D_t      = np.diag(dtanh[t])
        J_accum  = D_t @ Wh @ J_accum
        grad_norms[t] = np.linalg.norm(J_accum)

    return grad_norms


def spectral_radius(W: np.ndarray) -> float:
    """
    Compute the largest absolute eigenvalue of W.
    Controls whether gradients vanish (< 1) or explode (> 1).
    """
    eigenvalues = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigenvalues)))


# ---------------------------------------------------------------------------
# Parameter Initialisation
# ---------------------------------------------------------------------------

def init_rnn_params(
    d: int,
    n_h: int,
    n_out: int = 1,
    scale: float = 0.1,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Initialise RNN parameters with small random values.

    Parameters
    ----------
    d     : dimensionality of each input x_t
    n_h   : hidden state size
    n_out : output dimensionality
    scale : std-dev of random initialisation
    seed  : RNG seed

    Returns
    -------
    dict with keys: 'Wx', 'Wh', 'bh', 'Wy', 'by'
    """
    rng = np.random.default_rng(seed)
    return {
        'Wx': rng.standard_normal((n_h, d))     * scale,
        'Wh': rng.standard_normal((n_h, n_h))   * scale,
        'bh': np.zeros(n_h),
        'Wy': rng.standard_normal((n_out, n_h)) * scale,
        'by': np.zeros(n_out),
    }


def count_rnn_params(d: int, n_h: int, n_out: int = 1) -> Dict[str, int]:
    """
    Analytical parameter count for a vanilla RNN (no LSTM / GRU).

    Parameter groups
    ----------------
    Wx  : n_h × d
    Wh  : n_h × n_h
    bh  : n_h
    Wy  : n_out × n_h
    by  : n_out
    """
    Wx   = n_h * d
    Wh   = n_h * n_h
    bh   = n_h
    Wy   = n_out * n_h
    by_  = n_out
    total = Wx + Wh + bh + Wy + by_
    return {'Wx': Wx, 'Wh': Wh, 'bh': bh, 'Wy': Wy, 'by': by_, 'total': total}


# ---------------------------------------------------------------------------
# Gradient Clipping (NumPy version, mirrors PyTorch clip_grad_norm_)
# ---------------------------------------------------------------------------

def clip_gradient(grad: np.ndarray, max_norm: float = 5.0) -> Tuple[np.ndarray, float]:
    """
    Clip gradient vector if its L2 norm exceeds max_norm.

        g ← (max_norm / ||g||) * g    if ||g|| > max_norm

    Returns
    -------
    clipped_grad : same shape as grad
    norm         : original gradient norm (before clipping)
    """
    norm = float(np.linalg.norm(grad))
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad, norm


# ---------------------------------------------------------------------------
# Self-test / Worked Example from Slides
# ---------------------------------------------------------------------------

def slides_worked_example() -> None:
    """
    Reproduce the 2-step worked example from Lecture 12 slides exactly.

    Setup: d=1, n_h=2, tanh activation.
        Wx = [[0.5], [1.0]]
        Wh = [[0.5, 0], [0, 0.5]]
        bh = 0
        h0 = [0, 0]
        x(1) = 1,  x(2) = 2
    """
    Wx = np.array([[0.5], [1.0]])
    Wh = np.array([[0.5, 0.0], [0.0, 0.5]])
    bh = np.zeros(2)
    h0 = np.zeros(2)

    X  = np.array([[1.0], [2.0]])   # shape (2, 1) — 2 time steps, d=1
    H, Z = rnn_forward(X, Wx, Wh, bh, h0)

    print("=" * 60)
    print("SLIDES WORKED EXAMPLE (Section 3 — Elman RNN Forward Pass)")
    print("=" * 60)
    print(f"\n  Wx = {Wx.flatten()}")
    print(f"  Wh = {Wh}")
    print(f"  bh = {bh}")
    print(f"  h0 = {h0}")
    print(f"  Input: x(1) = 1, x(2) = 2")

    print(f"\n  Step 1 (t=1): x=1")
    print(f"    z(1) = Wh @ h0 + Wx @ 1 = {Z[0]}")
    print(f"    h(1) = tanh(z(1))        = {H[0].round(3)}")
    print(f"    Expected: [0.462, 0.762]")

    print(f"\n  Step 2 (t=2): x=2")
    print(f"    z(2) = Wh @ h(1) + Wx @ 2 = {Z[1].round(3)}")
    print(f"    h(2) = tanh(z(2))          = {H[1].round(3)}")
    print(f"    Expected: [0.843, 0.983]")

    # Gradient norms
    norms = compute_bptt_gradient_norms(Z, Wh)
    print(f"\n  Gradient norm back from t=2 to t=0: {norms}")
    print(f"  Spectral radius of Wh: {spectral_radius(Wh):.3f}")

    print("\n✅ All values match slides!\n")


if __name__ == "__main__":
    slides_worked_example()

    # Confirm step-by-step generator works
    Wx = np.array([[0.5], [1.0]])
    Wh = np.array([[0.5, 0.0], [0.0, 0.5]])
    bh = np.zeros(2)
    X  = np.array([[1.0], [2.0]])

    print("Step-by-step generator:")
    for step in rnn_forward_step_by_step(X, Wx, Wh, bh):
        t = step['t']
        print(f"  t={t}: h = {step['h_t'].round(3)}")

    # Parameter counting
    p = count_rnn_params(d=3, n_h=64, n_out=1)
    print(f"\nParam count (d=3, n_h=64, n_out=1): {p}")

    print("\n✅ rnn_core.py: all self-tests passed!")
