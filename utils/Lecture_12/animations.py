"""
animations.py — Matplotlib Animations for Lecture 12
=====================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides HTML-playable Matplotlib animations for use in Jupyter / Colab.
Each function returns a matplotlib.animation.FuncAnimation object that can be
displayed with IPython.display.HTML(anim.to_jshtml()).

Animations:
  1. animate_rnn_forward       — frame-by-frame hidden state propagation
  2. animate_bptt_gradient_flow — gradient signal flowing backwards in time
  3. animate_lstm_cell          — LSTM gate activations and cell state evolution
  4. animate_sequence_windowing — sliding window over a time series
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict


# ===========================================================================
# 1. RNN Forward Pass Animation
# ===========================================================================

def animate_rnn_forward(
    X       : np.ndarray,
    Wx      : np.ndarray,
    Wh      : np.ndarray,
    bh      : np.ndarray,
    h0      : Optional[np.ndarray] = None,
    figsize : tuple = (14, 5),
    interval: int = 700
) -> animation.FuncAnimation:
    """
    Animate the vanilla RNN forward pass step by step.

    Shows: input at each time step, current hidden state, and growing heatmap.

    Parameters
    ----------
    X        : (T, d)  — input sequence
    Wx,Wh,bh : RNN parameters
    h0       : initial hidden state (zeros if None)
    interval : milliseconds between frames

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
           Display with: HTML(anim.to_jshtml())
    """
    from utils.Lecture_12.rnn_core import rnn_forward_step_by_step

    T, d  = X.shape
    n_h   = Wh.shape[0]
    steps = list(rnn_forward_step_by_step(X, Wx, Wh, bh, h0))

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.5, 2])
    ax_input  = fig.add_subplot(gs[0])
    ax_hidden = fig.add_subplot(gs[1])
    ax_heat   = fig.add_subplot(gs[2])

    # --- Input axis ---
    ax_input.set_xlim(-0.5, d - 0.5)
    ax_input.set_ylim(-1.5, 1.5)
    ax_input.set_title('Input x(t)', fontsize=11, fontweight='bold')
    ax_input.set_xlabel('Feature dim', fontsize=10)
    bars_input = ax_input.bar(range(d), np.zeros(d), color='#3498db',
                               edgecolor='black')
    time_text = ax_input.text(0.5, 1.3, '', ha='center', fontsize=12,
                               fontweight='bold', transform=ax_input.transAxes)

    # --- Hidden state axis ---
    ax_hidden.set_xlim(-0.5, n_h - 0.5)
    ax_hidden.set_ylim(-1.1, 1.1)
    ax_hidden.set_title('Hidden State h(t)', fontsize=11, fontweight='bold')
    ax_hidden.set_xlabel('Hidden unit', fontsize=10)
    ax_hidden.axhline(0, color='black', linewidth=0.8)
    bars_hidden = ax_hidden.bar(range(n_h), np.zeros(n_h), color='#e74c3c',
                                 edgecolor='black')

    # --- Heatmap axis ---
    H_all = np.full((T, n_h), np.nan)
    im = ax_heat.imshow(H_all.T, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1, interpolation='nearest')
    ax_heat.set_title('Hidden State History', fontsize=11, fontweight='bold')
    ax_heat.set_xlabel('Time Step', fontsize=10)
    ax_heat.set_ylabel('Hidden Unit', fontsize=10)
    ax_heat.set_xticks(range(T))
    ax_heat.set_xticklabels([f't={i+1}' for i in range(T)], fontsize=8)
    plt.colorbar(im, ax=ax_heat, fraction=0.03)

    plt.tight_layout()

    def update(frame):
        step = steps[frame]
        t    = step['t']
        x_t  = step['x_t']
        h_t  = step['h_t']
        H_sf = step['H_so_far']

        # Update input bars
        for bar, val in zip(bars_input, x_t):
            bar.set_height(val)
            bar.set_facecolor('#3498db' if val >= 0 else '#c0392b')

        time_text.set_text(f't = {t+1}')

        # Update hidden state bars
        for bar, val in zip(bars_hidden, h_t):
            bar.set_height(val)
            bar.set_facecolor('#2ecc71' if val >= 0 else '#e74c3c')

        # Update heatmap
        H_all[:len(H_sf)] = H_sf
        im.set_data(H_all.T)

        return list(bars_input) + list(bars_hidden) + [im, time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=interval, blit=True, repeat=True
    )
    plt.close(fig)
    return anim


# ===========================================================================
# 2. BPTT Gradient Flow Animation
# ===========================================================================

def animate_bptt_gradient_flow(
    T       : int   = 20,
    rho     : float = 0.95,
    figsize : tuple = (13, 5),
    interval: int   = 300
) -> animation.FuncAnimation:
    """
    Animate the gradient signal propagating backwards through time during BPTT.

    Shows how the gradient norm shrinks (or explodes) as it travels further
    back in time from the loss at step T.

    Parameters
    ----------
    T       : sequence length
    rho     : spectral radius of W_h (controls vanishing/exploding)
    interval: ms between frames

    Returns
    -------
    FuncAnimation — display with HTML(anim.to_jshtml())
    """
    grad_norms = rho ** np.arange(T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ---- Left panel: gradient bar marching backward ----
    ax1.set_xlim(-1, T)
    ax1.set_ylim(0, max(grad_norms) * 1.2 + 1e-6)
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Gradient Norm', fontsize=11)
    ax1.set_title(f'BPTT: Gradient Propagating Backward\n(ρ = {rho:.2f})',
                  fontsize=11, fontweight='bold')
    ax1.axhline(0.1, linestyle='--', color='orange', alpha=0.7,
                label='Norm = 0.1 (near zero)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Pre-draw all bars in gray
    bars = ax1.bar(range(T), grad_norms,
                   color=['#bdc3c7'] * T, edgecolor='black', linewidth=0.6)

    current_marker = ax1.axvline(T - 1, color='#e74c3c', linewidth=3, alpha=0.9)
    step_text = ax1.text(0.5, 0.92, '', transform=ax1.transAxes,
                         ha='center', fontsize=12, fontweight='bold')

    # ---- Right panel: cumulative gradient trajectory ----
    ax2.set_xlim(-1, T)
    ax2.set_ylim(min(grad_norms) * 0.5, max(grad_norms) * 1.3)
    ax2.set_xlabel('Steps Back from Loss', fontsize=11)
    ax2.set_ylabel('Gradient Norm', fontsize=11)
    ax2.set_title('Gradient Norm Trajectory', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.axhline(1, linestyle='--', color='green', alpha=0.6, label='Norm = 1')
    ax2.legend(fontsize=9)

    traj_x, traj_y = [], []
    traj_line, = ax2.plot([], [], 'r-o', linewidth=2.5, markersize=6)

    def update(frame):
        # frame goes T-1 → 0 (backwards in time)
        step = T - 1 - frame
        current_marker.set_xdata([step, step])

        # Colour bars: already visited = coloured, not yet = gray
        for i, bar in enumerate(bars):
            if i >= step:
                norm = grad_norms[T - 1 - i]
                bar.set_facecolor('#e74c3c' if rho < 1 else '#e67e22')
                bar.set_alpha(0.7 + 0.3 * (i == step))
            else:
                bar.set_facecolor('#bdc3c7')

        step_text.set_text(f'Gradient at step {step} = {grad_norms[T-1-step]:.4f}')

        traj_x.append(frame)
        traj_y.append(grad_norms[T - 1 - step])
        traj_line.set_data(traj_x, traj_y)

        return list(bars) + [current_marker, traj_line, step_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=interval, blit=True, repeat=False
    )
    plt.close(fig)
    return anim


# ===========================================================================
# 3. LSTM Cell Animation
# ===========================================================================

def animate_lstm_cell(
    X       : np.ndarray,
    params  : Dict[str, np.ndarray],
    h0      : Optional[np.ndarray] = None,
    c0      : Optional[np.ndarray] = None,
    unit    : int = 0,
    figsize : tuple = (14, 6),
    interval: int = 600
) -> animation.FuncAnimation:
    """
    Animate LSTM gate activations and cell/hidden state evolving over a sequence.

    Parameters
    ----------
    X      : (T, d)   — input sequence
    params : dict from init_lstm_params()
    h0, c0 : initial states
    unit   : which hidden unit to visualise
    """
    from utils.Lecture_12.lstm_gru import lstm_forward

    H, C, Gates = lstm_forward(X, params, h0, c0)
    T  = X.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    plt.suptitle(f'LSTM Cell Evolution — Hidden Unit {unit}',
                 fontsize=12, fontweight='bold')

    gate_keys   = ['f', 'i', 'c_tilde', 'o']
    gate_titles = ['Forget Gate f', 'Input Gate i', 'Candidate c̃', 'Output Gate o']
    gate_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']

    # Top row: 4 gate axes
    gate_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]

    # Prepare lines for each gate
    gate_lines = []
    gate_t_data = {k: [] for k in gate_keys}
    gate_v_data = {k: [] for k in gate_keys}

    for ax, key, title, color in zip(gate_axes, gate_keys, gate_titles, gate_colors):
        ax.set_xlim(0, T)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axhline(0.5, linestyle='--', color='gray', alpha=0.4)
        ax.grid(True, alpha=0.3)
        line, = ax.plot([], [], color=color, linewidth=2.5)
        gate_lines.append(line)

    # Bottom middle: cell state
    ax_c = axes[1, 1]
    ax_c.set_xlim(0, T)
    ax_c.set_ylim(np.min(C[:, unit]) - 0.2, np.max(C[:, unit]) + 0.2)
    ax_c.set_title('Cell State c(t)', fontsize=10, fontweight='bold')
    ax_c.grid(True, alpha=0.3)
    line_c, = ax_c.plot([], [], '#9b59b6', linewidth=2.5)

    # Bottom right: hidden state
    ax_h = axes[1, 2]
    ax_h.set_xlim(0, T)
    ax_h.set_ylim(-1.1, 1.1)
    ax_h.set_title('Hidden State h(t)', fontsize=10, fontweight='bold')
    ax_h.grid(True, alpha=0.3)
    line_h, = ax_h.plot([], [], '#1abc9c', linewidth=2.5)

    t_data = []
    c_data = []
    h_data = []

    plt.tight_layout()

    def update(frame):
        t_data.append(frame)
        c_data.append(C[frame, unit])
        h_data.append(H[frame, unit])

        for line, key, (k_ti, k_vi) in zip(
            gate_lines, gate_keys,
            [(gate_t_data[k], gate_v_data[k]) for k in gate_keys]
        ):
            k_ti.append(frame)
            k_vi.append(Gates[key][frame, unit])
            line.set_data(k_ti, k_vi)

        line_c.set_data(t_data, c_data)
        line_h.set_data(t_data, h_data)

        return gate_lines + [line_c, line_h]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=interval, blit=True, repeat=True
    )
    plt.close(fig)
    return anim


# ===========================================================================
# 4. Sliding Window Animation
# ===========================================================================

def animate_sequence_windowing(
    data    : np.ndarray,
    lookback: int = 5,
    col     : int = 0,
    figsize : tuple = (13, 4),
    interval: int = 400
) -> animation.FuncAnimation:
    """
    Animate the sliding window being swept over a time series.

    Makes the concept of sequence windowing visually concrete.

    Parameters
    ----------
    data     : (N,) or (N, d) — time series
    lookback : L — window size
    col      : which column to plot if multivariate
    """
    if data.ndim == 1:
        ts = data
    else:
        ts = data[:, col]

    N = len(ts)
    n_windows = N - lookback

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(N), ts, 'k-o', linewidth=1.5, markersize=4,
            alpha=0.5, zorder=1, label='Time Series')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Sliding Window (Lookback L={lookback})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Window highlight
    win_line, = ax.plot([], [], 'b-o', linewidth=2.5, markersize=7,
                        zorder=2, label='Input Window')
    target_pt, = ax.plot([], [], 'r*', markersize=15, zorder=3, label='Target y')

    step_text = ax.text(0.02, 0.94, '', transform=ax.transAxes,
                        fontsize=11, va='top',
                        bbox=dict(boxstyle='round', facecolor='aliceblue'))
    ax.legend(fontsize=10)

    def update(frame):
        k = frame % n_windows
        win_x = list(range(k, k + lookback))
        win_y = ts[k : k + lookback]
        tgt_x = k + lookback
        tgt_y = ts[tgt_x]

        win_line.set_data(win_x, win_y)
        target_pt.set_data([tgt_x], [tgt_y])
        step_text.set_text(
            f'Window {k+1}/{n_windows}: steps [{k}..{k+lookback-1}] → target step {tgt_x}'
        )
        return win_line, target_pt, step_text

    anim = animation.FuncAnimation(
        fig, update, frames=n_windows, interval=interval, blit=True, repeat=True
    )
    plt.close(fig)
    return anim


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    from utils.Lecture_12.rnn_core  import init_rnn_params
    from utils.Lecture_12.lstm_gru  import init_lstm_params

    print("Testing animations.py...")

    # 1. RNN forward animation
    np.random.seed(42)
    T, d, n_h = 5, 2, 4
    p   = init_rnn_params(d, n_h)
    X   = np.random.randn(T, d)
    anim = animate_rnn_forward(X, p['Wx'], p['Wh'], p['bh'])
    print(f"  ✓ animate_rnn_forward: {anim.save_count} frames")

    # 2. BPTT gradient animation
    anim2 = animate_bptt_gradient_flow(T=15, rho=0.9)
    print(f"  ✓ animate_bptt_gradient_flow: {anim2.save_count} frames")

    # 3. LSTM cell animation
    lp = init_lstm_params(d, n_h)
    anim3 = animate_lstm_cell(X, lp)
    print(f"  ✓ animate_lstm_cell: {anim3.save_count} frames")

    # 4. Sequence windowing animation
    data = np.cumsum(np.random.randn(20))
    anim4 = animate_sequence_windowing(data, lookback=4)
    print(f"  ✓ animate_sequence_windowing: {anim4.save_count} frames")

    print("\n✅ animations.py: all self-tests passed!")
