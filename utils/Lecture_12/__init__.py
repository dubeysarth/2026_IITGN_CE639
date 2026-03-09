"""
Lecture 12: Recurrent Networks and Temporal Modeling
Helper utilities for CE 639: AI for Civil Engineering

This package provides:
  • From-scratch NumPy implementations of vanilla RNN, LSTM, and GRU
  • PyTorch model classes for training
  • Sequence data pipeline (windowing, batching, temporal splits)
  • Synthetic CE time-series datasets (streamflow, SHM, traffic, air quality, construction)
  • Rich visualisations, interactive ipywidgets, and Matplotlib animations
"""

# ── Core RNN Operations (NumPy, from scratch) ───────────────────────────────
from .rnn_core import (
    rnn_cell_forward,
    rnn_forward,
    rnn_forward_step_by_step,
    rnn_predict,
    compute_bptt_gradient_norms,
    spectral_radius,
    clip_gradient,
    init_rnn_params,
    count_rnn_params,
    slides_worked_example,
)

# ── LSTM & GRU Operations (NumPy, from scratch) ─────────────────────────────
from .lstm_gru import (
    sigmoid,
    tanh,
    lstm_cell_forward,
    lstm_forward,
    lstm_cell_gradient,
    gru_cell_forward,
    gru_forward,
    init_lstm_params,
    init_gru_params,
    count_params_lstm,
    count_params_gru,
    slides_lstm_worked_example,
)

# ── PyTorch Architectures ────────────────────────────────────────────────────
from .architectures import (
    SimpleRNN,
    SimpleLSTM,
    SimpleGRU,
    BidirectionalLSTM,
    StackedLSTM,
    Seq2SeqLSTM,
    count_parameters,
    model_summary,
    compare_architectures,
)

# ── Training Utilities ───────────────────────────────────────────────────────
from .training import (
    create_sequences,
    temporal_train_val_split,
    SequenceDataset,
    train_one_epoch,
    evaluate,
    train_rnn,
    nse_score,
    rmse,
    mae,
    plot_training_history,
)

# ── CE Datasets ──────────────────────────────────────────────────────────────
from .ce_datasets import (
    generate_streamflow,
    generate_vibration_signals,
    generate_shm_dataset,
    generate_traffic_data,
    generate_air_quality,
    generate_construction_progress,
)

# ── Visualisations ───────────────────────────────────────────────────────────
from .visualizations import (
    plot_hidden_state_heatmap,
    plot_gate_activations,
    plot_gradient_norms,
    plot_spectral_radius_demo,
    plot_sequence_types_diagram,
    plot_streamflow_prediction,
    plot_architecture_comparison_table,
    plot_parameter_comparison,
    plot_forecast_horizon,
    plot_lstm_vs_rnn_gradient_flow,
)

# ── Interactive Widgets ──────────────────────────────────────────────────────
from .widgets import (
    rnn_forward_widget,
    lstm_gate_widget,
    gradient_flow_widget,
    lookback_widget,
    architecture_comparison_widget,
)

# ── Animations ───────────────────────────────────────────────────────────────
from .animations import (
    animate_rnn_forward,
    animate_bptt_gradient_flow,
    animate_lstm_cell,
    animate_sequence_windowing,
)

__all__ = [
    # rnn_core
    'rnn_cell_forward', 'rnn_forward', 'rnn_forward_step_by_step',
    'rnn_predict', 'compute_bptt_gradient_norms', 'spectral_radius',
    'clip_gradient', 'init_rnn_params', 'count_rnn_params',
    'slides_worked_example',
    # lstm_gru
    'sigmoid', 'tanh', 'lstm_cell_forward', 'lstm_forward',
    'lstm_cell_gradient', 'gru_cell_forward', 'gru_forward',
    'init_lstm_params', 'init_gru_params', 'count_params_lstm',
    'count_params_gru', 'slides_lstm_worked_example',
    # architectures
    'SimpleRNN', 'SimpleLSTM', 'SimpleGRU', 'BidirectionalLSTM',
    'StackedLSTM', 'Seq2SeqLSTM', 'count_parameters', 'model_summary',
    'compare_architectures',
    # training
    'create_sequences', 'temporal_train_val_split', 'SequenceDataset',
    'train_one_epoch', 'evaluate', 'train_rnn', 'nse_score', 'rmse', 'mae',
    'plot_training_history',
    # ce_datasets
    'generate_streamflow', 'generate_vibration_signals', 'generate_shm_dataset',
    'generate_traffic_data', 'generate_air_quality', 'generate_construction_progress',
    # visualizations
    'plot_hidden_state_heatmap', 'plot_gate_activations', 'plot_gradient_norms',
    'plot_spectral_radius_demo', 'plot_sequence_types_diagram',
    'plot_streamflow_prediction', 'plot_architecture_comparison_table',
    'plot_parameter_comparison', 'plot_forecast_horizon',
    'plot_lstm_vs_rnn_gradient_flow',
    # widgets
    'rnn_forward_widget', 'lstm_gate_widget', 'gradient_flow_widget',
    'lookback_widget', 'architecture_comparison_widget',
    # animations
    'animate_rnn_forward', 'animate_bptt_gradient_flow',
    'animate_lstm_cell', 'animate_sequence_windowing',
]
