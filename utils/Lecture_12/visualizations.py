"""
visualizations.py — Rich Plotting Functions for Lecture 12
===========================================================
CE 639: AI for Civil Engineering — Lecture 12

Provides all visualisation helpers used by the main notebook:
  • Hidden-state heatmaps
  • LSTM gate activation plots
  • Gradient norm flow (vanishing/exploding demo)
  • Spectral radius decay/explosion demo
  • Sequence types diagram
  • RNN vs LSTM gradient comparison
  • Streamflow prediction overlay
  • Architecture comparison table
  • Parameter count bar chart
  • Forecast horizon degradation plot
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Tuple, Dict


# ===========================================================================
# 1. Hidden State Heatmap
# ===========================================================================

def plot_hidden_state_heatmap(
    H      : np.ndarray,
    title  : str = 'Hidden State over Time',
    figsize: Tuple = (14, 4),
    cmap   : str = 'RdBu_r'
) -> Tuple:
    """
    Heatmap of hidden states H : (T, n_h).

    Each row = one hidden dimension; each column = one time step.
    Reveals which hidden units activate and for how long.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(H.T, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Activation')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Hidden Unit Index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, ax


# ===========================================================================
# 2. LSTM Gate Activations
# ===========================================================================

def plot_gate_activations(
    gates_over_time: Dict[str, np.ndarray],
    time_index     : Optional[np.ndarray] = None,
    unit           : int = 0,
    figsize        : Tuple = (14, 8)
) -> Tuple:
    """
    Plot forget / input / output gate values over the sequence for one hidden unit.

    Parameters
    ----------
    gates_over_time : dict from lstm_forward() — keys: 'f','i','c_tilde','o','c','h'
    time_index      : time axis labels (e.g. dates)
    unit            : which hidden unit (dimension) to plot
    figsize         : figure dimensions
    """
    keys   = ['f', 'i', 'c_tilde', 'o', 'c', 'h']
    titles = ['Forget Gate $f^{(t)}$', 'Input Gate $i^{(t)}$',
              'Candidate $\\tilde{c}^{(t)}$', 'Output Gate $o^{(t)}$',
              'Cell State $c^{(t)}$', 'Hidden State $h^{(t)}$']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

    T = gates_over_time['f'].shape[0]
    t = time_index if time_index is not None else np.arange(T)

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for i, (key, title, color) in enumerate(zip(keys, titles, colors)):
        vals = gates_over_time[key][:, unit] if gates_over_time[key].ndim > 1 \
               else gates_over_time[key]
        axes[i].plot(t, vals, color=color, linewidth=2)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Value', fontsize=10)

        # Reference lines for gate activations
        if key in ('f', 'i', 'o'):
            axes[i].axhline(0.5, linestyle='--', color='gray', alpha=0.5)
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].fill_between(t, 0.5, vals, where=vals > 0.5,
                                 alpha=0.2, color=color)

    axes[-2].set_xlabel('Time Step', fontsize=10)
    axes[-1].set_xlabel('Time Step', fontsize=10)

    plt.suptitle(f'LSTM Gate Activations (Hidden Unit {unit})',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, axes


# ===========================================================================
# 3. Gradient Norm Flow (Vanishing / Exploding)
# ===========================================================================

def plot_gradient_norms(
    grad_norms      : np.ndarray,
    title           : str = 'Gradient Norm vs. Time Step',
    label           : str = 'Vanilla RNN',
    second_norms    : Optional[np.ndarray] = None,
    second_label    : str = 'LSTM',
    figsize         : Tuple = (12, 5)
) -> Tuple:
    """
    Bar / line plot of gradient norms at each time step.

    Visualises the vanishing (norms → 0) or exploding (norms → ∞) problem.
    Optionally overlays a second model (e.g. LSTM) for comparison.
    """
    T = len(grad_norms)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogy(t, grad_norms + 1e-12, 'r-o', linewidth=2,
                markersize=5, label=label, zorder=3)

    if second_norms is not None:
        ax.semilogy(t, second_norms + 1e-12, 'b-s', linewidth=2,
                    markersize=5, label=second_label, zorder=3)

    ax.axhline(1.0, linestyle='--', color='green', linewidth=1.5,
               label='Norm = 1 (no vanishing)', alpha=0.7)

    ax.set_xlabel('Time Step (backward from loss)', fontsize=12)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.invert_xaxis()   # step 0=earliest in history, T-1=closest to loss

    plt.tight_layout()
    return fig, ax


# ===========================================================================
# 4. Spectral Radius Demo
# ===========================================================================

def plot_spectral_radius_demo(
    radii   : List[float] = [0.5, 0.9, 0.99, 1.01, 1.1, 2.0],
    T_steps : int = 50,
    figsize : Tuple = (14, 6)
) -> Tuple:
    """
    Plot gradient norm decay/explosion for different spectral radii of W_h.

    grad_norm(t) ≈ rho^t  where rho = spectral radius of W_h.

    This is the core intuition for vanishing vs exploding gradients.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    t = np.arange(T_steps)

    cmap = plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(radii)))

    for rho, color in zip(radii, colors):
        norms = np.array([rho ** k for k in t])
        label = f'ρ = {rho}'
        ax1.plot(t, norms, linewidth=2, label=label, color=color)
        ax2.semilogy(t, norms + 1e-15, linewidth=2, label=label, color=color)

    for ax in (ax1, ax2):
        ax.axhline(1, linestyle='--', color='black', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Steps Back in Time', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('Gradient Norm', fontsize=11)
    ax1.set_title('Linear Scale', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.1, min(5, max(rho**T_steps for rho in radii) + 1))

    ax2.set_ylabel('Gradient Norm (log)', fontsize=11)
    ax2.set_title('Log Scale', fontsize=12, fontweight='bold')

    plt.suptitle('Spectral Radius ρ(W_h) Controls Gradient Flow\n'
                 'ρ < 1 → Vanishing   |   ρ > 1 → Exploding',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, (ax1, ax2)


# ===========================================================================
# 5. Sequence Types Diagram
# ===========================================================================

def plot_sequence_types_diagram(figsize: Tuple = (14, 6)) -> Tuple:
    """
    Draw the canonical sequence-to-sequence problem types diagram:
    one-to-one, one-to-many, many-to-one, many-to-many.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    configs = [
        {'title': 'One-to-One\n(Feedforward NN)',
         'inputs': [1], 'outputs': [1], 'ce': 'Beam deflection'},
        {'title': 'One-to-Many\n(Image Caption)',
         'inputs': [1], 'outputs': [1, 2, 3], 'ce': 'Rainfall scenario gen.'},
        {'title': 'Many-to-One\n(Classification)',
         'inputs': [1, 2, 3], 'outputs': [3], 'ce': 'SHM damage detection'},
        {'title': 'Many-to-Many\n(Seq2Seq)',
         'inputs': [1, 2, 3], 'outputs': [1, 2, 3], 'ce': 'Streamflow forecast'},
    ]

    BOX = dict(boxstyle='round', facecolor='#3498db', alpha=0.85)
    OUT_BOX = dict(boxstyle='round', facecolor='#e74c3c', alpha=0.85)

    for ax, cfg in zip(axes, configs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title(cfg['title'], fontsize=10, fontweight='bold', pad=8)

        n_in  = len(cfg['inputs'])
        n_out = len(cfg['outputs'])

        # Draw input boxes (blue)
        for i, pos in enumerate(cfg['inputs']):
            y = 3.5 + i * 0.7
            ax.text(0.25, y, f'x_{pos}', ha='center', va='center',
                    fontsize=11, color='white', bbox=BOX, fontweight='bold')

        # Arrow
        ax.annotate('', xy=(0.75, 3.5),
                    xytext=(0.25 + 0.05, 3.5 + (n_in - 1) * 0.35),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

        # Draw output boxes (red)
        for i, pos in enumerate(cfg['outputs']):
            y = 3.5 + i * 0.7
            ax.text(0.75, y, f'y_{pos}', ha='center', va='center',
                    fontsize=11, color='white', bbox=OUT_BOX, fontweight='bold')

        # CE example label
        ax.text(0.5, 1.2, f'CE: {cfg["ce"]}',
                ha='center', va='center', fontsize=8.5,
                style='italic', color='#555')

    plt.suptitle('Sequence Problem Types', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, axes


# ===========================================================================
# 6. Streamflow Prediction Plot
# ===========================================================================

def plot_streamflow_prediction(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    time_index : Optional[np.ndarray] = None,
    nse        : Optional[float] = None,
    title      : str = 'Streamflow Prediction — Observed vs LSTM',
    figsize    : Tuple = (14, 5)
) -> Tuple:
    """
    Plot observed vs predicted streamflow hydrograph.

    Parameters
    ----------
    y_true     : (N,) — observed streamflow
    y_pred     : (N,) — predicted streamflow
    time_index : (N,) — optional time axis
    nse        : NSE score to annotate
    """
    t = time_index if time_index is not None else np.arange(len(y_true))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # ---- Hydrograph ----
    ax1.fill_between(t, y_true, alpha=0.3, color='#3498db', label='Observed')
    ax1.plot(t, y_true, '#3498db', linewidth=1.5, alpha=0.8)
    ax1.plot(t, y_pred, '#e74c3c', linewidth=2, linestyle='-',
             label='LSTM Predicted', zorder=3)
    ax1.set_ylabel('Streamflow (m³/s)', fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    if nse is not None:
        ax1.text(0.02, 0.95, f'NSE = {nse:.3f}',
                 transform=ax1.transAxes, fontsize=12, fontweight='bold',
                 va='top', color='#27ae60' if nse > 0.7 else '#c0392b',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---- Residuals ----
    residuals = y_true - y_pred
    ax2.bar(t, residuals, color=np.where(residuals >= 0, '#3498db', '#e74c3c'),
            alpha=0.6, width=1.0)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel('Time Step (days)', fontsize=11)
    ax2.set_ylabel('Residual', fontsize=11)
    ax2.set_title('Prediction Residuals', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


# ===========================================================================
# 7. Architecture Comparison Table
# ===========================================================================

def plot_architecture_comparison_table(figsize: Tuple = (12, 5)) -> Tuple:
    """
    Draw the FNN vs CNN vs RNN/LSTM comparison table from the slides.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    columns = ['Criterion', 'Feedforward NN', 'CNN', 'RNN / LSTM']
    rows = [
        ['Data type',        'Tabular',        'Spatial grid',    'Ordered sequence'],
        ['Order matters?',   'No',             'Local (2D)',       'Yes'],
        ['Long-range dep.',  'No',             'Limited',         'Yes'],
        ['Variable length',  'No',             'No',              'Yes'],
        ['CE example',       'Beam deflection','Crack detection', 'Streamflow'],
    ]

    row_colors = [['#ecf0f1'] * 4, ['#ffffff'] * 4] * 3
    cell_colors = [
        ['#2c3e50', '#3498db', '#e74c3c', '#27ae60'],   # header
    ] + row_colors[:len(rows)]

    col_widths = [0.22, 0.22, 0.22, 0.24]
    table = ax.table(
        cellText = rows,
        colLabels= columns,
        cellLoc  = 'center',
        loc      = 'center',
        colWidths= col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(4):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight last column (RNN)
    for i in range(1, len(rows) + 1):
        table[i, 3].set_facecolor('#d5f5e3')

    ax.set_title('When to Use Which Architecture', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig, ax


# ===========================================================================
# 8. Parameter Count Comparison
# ===========================================================================

def plot_parameter_comparison(
    d   : int = 3,
    n_h : int = 64,
    figsize: Tuple = (10, 6)
) -> Tuple:
    """
    Bar chart comparing parameter counts: Vanilla RNN vs LSTM vs GRU.
    """
    from utils.Lecture_12.lstm_gru import count_params_lstm, count_params_gru
    from utils.Lecture_12.rnn_core import count_rnn_params

    p_rnn  = count_rnn_params(d, n_h)['total']
    p_lstm = count_params_lstm(d, n_h)['total']
    p_gru  = count_params_gru(d, n_h)['total']

    models = ['Vanilla RNN', 'LSTM', 'GRU']
    params = [p_rnn, p_lstm, p_gru]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(models, params, color=colors, edgecolor='black', linewidth=1.5)

    for bar, p in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{p:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title(f'Parameter Comparison (d={d}, n_h={n_h})',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


# ===========================================================================
# 9. Forecast Horizon Degradation
# ===========================================================================

def plot_forecast_horizon(
    y_true   : np.ndarray,
    y_preds  : Dict[int, np.ndarray],
    metric_fn= None,
    figsize  : Tuple = (12, 5)
) -> Tuple:
    """
    Plot prediction quality vs forecast horizon.

    Parameters
    ----------
    y_true  : (N,)               — ground truth
    y_preds : {horizon: y_pred}  — predictions at each horizon
    metric_fn : callable(y_true, y_pred) → float   default: NSE
    """
    if metric_fn is None:
        from utils.Lecture_12.training import nse_score
        metric_fn = nse_score
        metric_name = 'NSE'
    else:
        metric_name = 'Score'

    horizons = sorted(y_preds.keys())
    scores   = [metric_fn(y_true, y_preds[h]) for h in horizons]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(horizons, scores, 'o-', linewidth=2.5, markersize=8, color='#3498db')
    ax.axhline(0.7, linestyle='--', color='#27ae60', linewidth=1.5,
               label='NSE = 0.7 ("Good" threshold)')
    ax.set_xlabel('Forecast Horizon (steps)', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title('Forecast Quality vs. Horizon', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ===========================================================================
# 10. LSTM vs RNN Gradient Comparison (side-by-side)
# ===========================================================================

def plot_lstm_vs_rnn_gradient_flow(
    T       : int   = 60,
    rho     : float = 0.95,
    f_gate  : float = 0.9,
    figsize : Tuple = (13, 5)
) -> Tuple:
    """
    Side-by-side gradient norm comparison: Vanilla RNN vs LSTM.

    RNN  gradient decays as rho^t
    LSTM gradient decays as prod(f_gate)^t ≈ f_gate^t
    (forget gate ≈ 1 → gradient can flow further)

    Parameters
    ----------
    rho    : spectral radius of W_h (RNN)
    f_gate : approximate constant forget gate value (LSTM)
    """
    t = np.arange(T)

    rnn_grads  = rho   ** t
    lstm_grads = f_gate ** t   # forget-gate product approximation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, grads, name, color in zip(
        (ax1, ax2),
        (rnn_grads, lstm_grads),
        (f'Vanilla RNN (ρ={rho})', f'LSTM (f≈{f_gate})'),
        ('#e74c3c', '#2ecc71')
    ):
        ax.fill_between(t, grads, alpha=0.3, color=color)
        ax.plot(t, grads, color=color, linewidth=2.5)
        ax.set_xlabel('Time Steps Back', fontsize=11)
        ax.set_ylabel('Gradient Norm', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axhline(0.5, linestyle='--', color='gray', alpha=0.6, label='Norm = 0.5')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Gradient Highway: RNN vs LSTM\n'
                 'How far back in time can each model learn from the past?',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, (ax1, ax2)


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # non-interactive for self-test

    print("Testing visualizations.py...")

    # Hidden state heatmap
    H = np.random.randn(30, 16)
    fig, _ = plot_hidden_state_heatmap(H)
    plt.close(fig)
    print("  ✓ plot_hidden_state_heatmap")

    # Gate activations (mock gates)
    T_test = 30
    gates = {k: np.random.rand(T_test, 8) for k in ('f','i','c_tilde','o','c','h')}
    fig, _ = plot_gate_activations(gates)
    plt.close(fig)
    print("  ✓ plot_gate_activations")

    # Gradient norms
    norms = np.exp(-np.linspace(0, 5, 30))
    fig, _ = plot_gradient_norms(norms)
    plt.close(fig)
    print("  ✓ plot_gradient_norms")

    # Spectral radius demo
    fig, _ = plot_spectral_radius_demo()
    plt.close(fig)
    print("  ✓ plot_spectral_radius_demo")

    # Sequence types diagram
    fig, _ = plot_sequence_types_diagram()
    plt.close(fig)
    print("  ✓ plot_sequence_types_diagram")

    # Streamflow prediction
    y_true = np.abs(np.random.randn(100)) * 20 + 10
    y_pred = y_true + np.random.randn(100) * 2
    fig, _ = plot_streamflow_prediction(y_true, y_pred, nse=0.92)
    plt.close(fig)
    print("  ✓ plot_streamflow_prediction")

    # Architecture comparison
    fig, _ = plot_architecture_comparison_table()
    plt.close(fig)
    print("  ✓ plot_architecture_comparison_table")

    # LSTM vs RNN gradient flow
    fig, _ = plot_lstm_vs_rnn_gradient_flow()
    plt.close(fig)
    print("  ✓ plot_lstm_vs_rnn_gradient_flow")

    print("\n✅ visualizations.py: all self-tests passed!")
