"""
widgets.py — Interactive ipywidgets Explorers for Lecture 12
============================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides five interactive explorers callable in a Jupyter / Colab environment.
Each function is self-contained and includes a graceful fallback message
when ipywidgets or PyTorch is unavailable.

Widgets:
  1. rnn_forward_widget         — step through RNN forward pass
  2. lstm_gate_widget           — adjust inputs and see gate values
  3. gradient_flow_widget       — vary spectral radius and see gradient decay
  4. lookback_widget            — vary lookback L on streamflow data
  5. architecture_comparison_widget — compare RNN/LSTM/GRU counts and curves
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def rnn_forward_widget() -> None:
    """
    Interactive step-through of the vanilla RNN forward pass.

    Sliders control: Wx values, Wh diagonal, x1, x2, x3 inputs.
    Shows per-step hidden state and accumulating hidden state heatmap.
    """
    try:
        from ipywidgets import interact, widgets, interactive_output, VBox, HBox, HTML
        import ipywidgets as ipw
    except ImportError:
        print("⚠️  ipywidgets not available — run in Jupyter/Colab.")
        print("    Install: pip install ipywidgets")
        return

    from utils.Lecture_12.rnn_core import rnn_forward

    def update(wx1=0.5, wx2=1.0, wh_diag=0.5, x1=1.0, x2=2.0, x3=0.5):
        Wx = np.array([[wx1], [wx2]])
        Wh = np.diag([wh_diag, wh_diag])
        bh = np.zeros(2)
        X  = np.array([[x1], [x2], [x3]])

        H, Z = rnn_forward(X, Wx, Wh, bh)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Hidden state per step
        for t in range(3):
            axes[0].bar([t - 0.2, t + 0.2], H[t],
                        width=0.35, color=['#3498db', '#e74c3c'],
                        label=['h[0]', 'h[1]'] if t == 0 else ['', ''])
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_xticklabels(['t=1', 't=2', 't=3'])
        axes[0].set_ylabel('Hidden State Value', fontsize=11)
        axes[0].set_title('Hidden State at Each Time Step', fontsize=11, fontweight='bold')
        axes[0].legend(['h₀', 'h₁'], fontsize=10)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].axhline(0, color='black', linewidth=0.8)
        axes[0].grid(True, alpha=0.3)

        # Heatmap
        im = axes[1].imshow(H.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_xticks([0, 1, 2])
        axes[1].set_xticklabels(['t=1', 't=2', 't=3'])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['h₀', 'h₁'])
        axes[1].set_title('Hidden State Heatmap', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=axes[1])

        plt.suptitle(
            f"Wx=[{wx1:.2f},{wx2:.2f}], Wh_diag={wh_diag:.2f}, "
            f"X=[{x1:.1f},{x2:.1f},{x3:.1f}]",
            fontsize=10
        )
        plt.tight_layout()
        plt.show()

        print(f"\n  h(1) = {H[0].round(3)}")
        print(f"  h(2) = {H[1].round(3)}")
        print(f"  h(3) = {H[2].round(3)}")

    interact(
        update,
        wx1    = widgets.FloatSlider(min=-2,  max=2, step=0.1,  value=0.5,  description='Wx[0]:'),
        wx2    = widgets.FloatSlider(min=-2,  max=2, step=0.1,  value=1.0,  description='Wx[1]:'),
        wh_diag= widgets.FloatSlider(min=-1,  max=1, step=0.05, value=0.5,  description='Wh diag:'),
        x1     = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=1.0,  description='x(1):'),
        x2     = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=2.0,  description='x(2):'),
        x3     = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=0.5,  description='x(3):'),
    )


def lstm_gate_widget() -> None:
    """
    Interactive single-step LSTM gate explorer.

    Sliders: h_prev, x_t, c_prev, and four weight scalars.
    Live output: gate values (f, i, c_tilde, o), c_t, h_t.
    Reproduces the spirit of the slides' scalar LSTM example.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("⚠️  ipywidgets not available — run in Jupyter/Colab.")
        return

    from utils.Lecture_12.lstm_gru import (
        lstm_cell_forward, sigmoid, tanh
    )

    def update(h_prev=0.5, x_t=1.0, c_prev=0.8,
               wf0=-1.0, wf1=2.0, wi0=1.0, wi1=1.0):
        # Scalar LSTM (n_h=1, d=1) — mirrors practice problem
        Wf = np.array([[wf0, wf1]])
        Wi = np.array([[wi0, wi1]])
        Wc = np.array([[ 0.5, -0.5]])   # fixed as in example
        Wo = np.array([[ 1.0,  0.0]])

        bf = bi = bc = bo = np.zeros(1)

        h_t, c_t, gates = lstm_cell_forward(
            np.array([x_t]), np.array([h_prev]), np.array([c_prev]),
            Wf, Wi, Wc, Wo, bf, bi, bc, bo
        )

        f   = gates['f'][0]
        i   = gates['i'][0]
        ct  = gates['c_tilde'][0]
        o   = gates['o'][0]
        c_new = c_t[0]
        h_new = h_t[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # Gate bar plot
        gate_vals  = [f, i, ct, o]
        gate_names = ['Forget f', 'Input i', 'Candidate c̃', 'Output o']
        colors     = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
        bars = ax1.bar(gate_names, gate_vals, color=colors, edgecolor='black', linewidth=1.5)
        ax1.axhline(0, color='black', linewidth=0.8)
        ax1.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_title('LSTM Gate Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Activation', fontsize=11)
        ax1.grid(True, alpha=0.3)

        for bar, v in zip(bars, gate_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.05 * np.sign(v),
                     f'{v:.3f}', ha='center', va='center', fontsize=11, fontweight='bold')

        # State update diagram
        ax2.axis('off')
        info = (
            f"  h(t-1) = {h_prev:.3f}\n"
            f"  x(t)   = {x_t:.3f}\n"
            f"  c(t-1) = {c_prev:.3f}\n"
            f"\n  ──────────────────\n"
            f"  Forget gate  f = {f:.3f}  ({'retain' if f > 0.5 else 'erase'})\n"
            f"  Input  gate  i = {i:.3f}\n"
            f"  Candidate    c̃ = {ct:.3f}\n"
            f"  Output gate  o = {o:.3f}\n"
            f"\n  Cell update:\n"
            f"  c(t) = f⊙c_prev + i⊙c̃ = {f:.2f}×{c_prev:.2f} + {i:.2f}×{ct:.2f}\n"
            f"       = {c_new:.4f}\n"
            f"\n  h(t) = o⊙tanh(c(t))\n"
            f"       = {o:.3f}×{np.tanh(c_new):.3f} = {h_new:.4f}\n"
        )
        ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
                 fontsize=11, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa'))

        plt.suptitle('Interactive LSTM Cell (Scalar, n_h=1, d=1)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    interact(
        update,
        h_prev = widgets.FloatSlider(min=-1,  max=1, step=0.05, value=0.5,  description='h(t-1):'),
        x_t    = widgets.FloatSlider(min=-3,  max=3, step=0.1,  value=1.0,  description='x(t):'),
        c_prev = widgets.FloatSlider(min=-2,  max=2, step=0.1,  value=0.8,  description='c(t-1):'),
        wf0    = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=-1.0, description='w_f[0]:'),
        wf1    = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=2.0,  description='w_f[1]:'),
        wi0    = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=1.0,  description='w_i[0]:'),
        wi1    = widgets.FloatSlider(min=-3,  max=3, step=0.5,  value=1.0,  description='w_i[1]:'),
    )


def gradient_flow_widget() -> None:
    """
    Gradient flow visualiser: vary spectral radius and sequence length.

    Shows RNN gradient norm decay/explosion vs LSTM (forget gate = const).
    Makes the vanishing gradient problem viscerally clear.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("⚠️  ipywidgets not available — run in Jupyter/Colab.")
        return

    def update(rho=0.95, f_gate=0.9, T=50):
        t          = np.arange(T)
        rnn_norms  = rho    ** t
        lstm_norms = f_gate ** t

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        ax1.fill_between(t, rnn_norms,  alpha=0.3, color='#e74c3c')
        ax1.plot(t, rnn_norms,  '#e74c3c', linewidth=2.5, label=f'Vanilla RNN (ρ={rho:.2f})')
        ax1.fill_between(t, lstm_norms, alpha=0.3, color='#2ecc71')
        ax1.plot(t, lstm_norms, '#2ecc71', linewidth=2.5, label=f'LSTM (f≈{f_gate:.2f})')
        ax1.set_ylim(-0.05, 1.1)
        ax1.set_xlabel('Time Steps Back from Loss', fontsize=11)
        ax1.set_ylabel('Gradient Norm', fontsize=11)
        ax1.set_title('Linear Scale', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(t, rnn_norms  + 1e-15, '#e74c3c', linewidth=2.5,
                     label=f'RNN (ρ={rho:.2f})')
        ax2.semilogy(t, lstm_norms + 1e-15, '#2ecc71', linewidth=2.5,
                     label=f'LSTM (f≈{f_gate:.2f})')
        ax2.axhline(1, linestyle='--', color='gray', alpha=0.5)
        ax2.set_xlabel('Time Steps Back', fontsize=11)
        ax2.set_ylabel('Gradient Norm (log)', fontsize=11)
        ax2.set_title('Log Scale', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        # Annotation
        half_life_rnn  = int(0.5 / max(1e-9, np.log(1/(rho+1e-9)))) \
                          if rho < 1 else T
        half_life_lstm = int(0.5 / max(1e-9, np.log(1/(f_gate+1e-9)))) \
                          if f_gate < 1 else T

        plt.suptitle(
            f'Gradient Reaches 0.5× at: RNN → step {min(half_life_rnn, T)}, '
            f'LSTM → step {min(half_life_lstm, T)}',
            fontsize=11, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()

    interact(
        update,
        rho    = widgets.FloatSlider(min=0.5, max=1.2, step=0.02, value=0.95,
                                     description='RNN ρ(W_h):',
                                     style={'description_width': 'initial'}),
        f_gate = widgets.FloatSlider(min=0.5, max=1.0, step=0.02, value=0.9,
                                     description='LSTM f gate:',
                                     style={'description_width': 'initial'}),
        T      = widgets.IntSlider(min=10, max=100, step=5, value=50,
                                   description='Sequence Length:',
                                   style={'description_width': 'initial'}),
    )


def lookback_widget() -> None:
    """
    Effect of lookback window size on streamflow prediction quality.

    Uses a synthetic LSTM (random weights) to show how NSE changes
    as lookback varies from 1 to 60 days.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("⚠️  ipywidgets not available — run in Jupyter/Colab.")
        return

    from utils.Lecture_12.ce_datasets import generate_streamflow
    from utils.Lecture_12.training import create_sequences, nse_score

    # Pre-generate data once
    data, t_days, meta = generate_streamflow(n_days=365)
    target_col = meta['target_col']

    def update(lookback=30, noise_level=0.2):
        try:
            X, y = create_sequences(data, lookback=lookback,
                                    horizon=1, target_col=target_col)
        except ValueError:
            print(f"Lookback {lookback} too large for 365 days of data")
            return

        # Simulate "prediction" as smoothed target + noise (stand-in for a trained model)
        np.random.seed(42)
        y_pred  = y + noise_level * np.std(y) * np.random.randn(len(y))
        nse_val = nse_score(y, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        # Hydrograph
        t_plot = np.arange(len(y))
        ax1.plot(t_plot, y,      '#3498db', linewidth=1.5, alpha=0.8, label='Observed Q')
        ax1.plot(t_plot, y_pred, '#e74c3c', linewidth=2, linestyle='--', label='Predicted Q')
        ax1.set_xlabel('Time (days)', fontsize=11)
        ax1.set_ylabel('Streamflow (m³/s)', fontsize=11)
        ax1.set_title(f'Streamflow (Lookback = {lookback} days)', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.02, 0.95, f'NSE = {nse_val:.3f}',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold',
                 va='top', color='#27ae60' if nse_val > 0.7 else '#c0392b',
                 bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))

        # Sliding window illustration
        ax2.imshow(X[:10, :, target_col], aspect='auto', cmap='Blues',
                   interpolation='nearest')
        ax2.set_xlabel('Lookback Days', fontsize=11)
        ax2.set_ylabel('Window Index', fontsize=11)
        ax2.set_title(f'Input Windows (first 10), L={lookback}',
                      fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()

        print(f"\n  Lookback = {lookback} days → {len(X)} windows")
        print(f"  NSE = {nse_val:.4f}")

    interact(
        update,
        lookback    = widgets.IntSlider(min=1, max=60, step=1, value=30,
                                        description='Lookback L:',
                                        style={'description_width': 'initial'}),
        noise_level = widgets.FloatSlider(min=0.01, max=0.5, step=0.02, value=0.2,
                                          description='Noise σ:',
                                          style={'description_width': 'initial'}),
    )


def architecture_comparison_widget() -> None:
    """
    Compare RNN vs LSTM vs GRU parameter counts and simulated loss curves.

    Sliders: hidden_size, num_layers.
    Shows: parameter count bar chart + simulated training curves.
    """
    try:
        from ipywidgets import interact, widgets
    except ImportError:
        print("⚠️  ipywidgets not available — run in Jupyter/Colab.")
        return

    from utils.Lecture_12.rnn_core  import count_rnn_params
    from utils.Lecture_12.lstm_gru  import count_params_lstm, count_params_gru

    def update(d=3, hidden_size=64, num_layers=1, show_curves=True):
        # Parameter counts (analytical, per layer × num_layers)
        rnn_p  = count_rnn_params(d, hidden_size)['total']  * num_layers
        lstm_p = count_params_lstm(d, hidden_size)['total'] * num_layers
        gru_p  = count_params_gru(d, hidden_size)['total']  * num_layers

        fig, axes = plt.subplots(1, 2 if show_curves else 1, figsize=(13, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # Parameter bar chart
        ax = axes[0]
        bars = ax.bar(['Vanilla RNN', 'LSTM', 'GRU'],
                      [rnn_p, lstm_p, gru_p],
                      color=['#3498db', '#e74c3c', '#2ecc71'],
                      edgecolor='black', linewidth=1.5)
        for bar, p in zip(bars, [rnn_p, lstm_p, gru_p]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{p:,}', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel('Parameters', fontsize=11)
        ax.set_title(f'Params (d={d}, n_h={hidden_size}, layers={num_layers})',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Simulated loss curves
        if show_curves and len(axes) > 1:
            ax2 = axes[1]
            ep  = np.arange(1, 31)
            np.random.seed(42)

            for name, color, scale in [
                ('Vanilla RNN', '#3498db', 1.3),
                ('LSTM',        '#e74c3c', 1.0),
                ('GRU',         '#2ecc71', 1.05),
            ]:
                loss = 0.8 * np.exp(-ep / (8 * scale)) + 0.05 + \
                       0.02 * np.random.randn(len(ep))
                ax2.plot(ep, np.clip(loss, 0.02, None), linewidth=2.5,
                         color=color, label=name)

            ax2.set_xlabel('Epoch', fontsize=11)
            ax2.set_ylabel('Validation Loss (simulated)', fontsize=11)
            ax2.set_title('Training Dynamics (schematic)', fontsize=11, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    interact(
        update,
        d           = widgets.IntSlider(min=1, max=10, value=3,
                                        description='Features d:',
                                        style={'description_width': 'initial'}),
        hidden_size = widgets.Dropdown(options=[16, 32, 64, 128, 256],
                                       value=64, description='Hidden n_h:',
                                       style={'description_width': 'initial'}),
        num_layers  = widgets.IntSlider(min=1, max=3, value=1,
                                        description='Layers:',
                                        style={'description_width': 'initial'}),
        show_curves = widgets.Checkbox(value=True, description='Show loss curves'),
    )


if __name__ == "__main__":
    print("Widget utilities for Lecture 12 loaded.")
    print("Run in Jupyter/Colab to use interactive explorers:")
    print("  rnn_forward_widget()")
    print("  lstm_gate_widget()")
    print("  gradient_flow_widget()")
    print("  lookback_widget()")
    print("  architecture_comparison_widget()")
