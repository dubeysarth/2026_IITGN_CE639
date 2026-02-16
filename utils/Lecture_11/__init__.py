"""
Lecture 11: Convolutional Neural Networks
Helper utilities for CE 639: AI for Civil Engineering

This package provides from-scratch implementations of CNN operations
and helper functions for visualization, training, and CE applications.
"""

# Core operations
from .convolution import (
    conv1d,
    conv2d,
    conv2d_multichannel,
    output_size,
    conv2d_step_by_step,
    common_kernels
)

from .pooling import (
    max_pool2d,
    avg_pool2d,
    global_avg_pool,
    pool_step_by_step
)

from .architectures import (
    SimpleCNN,
    LeNet5,
    make_vgg_block,
    ResidualBlock,
    SimpleResNet,
    count_parameters,
    model_summary
)

from .training import (
    train_one_epoch,
    evaluate,
    train_cnn,
    plot_training_history,
    get_predictions
)

from .augmentation import (
    augment_numpy,
    augmentation_gallery,
    get_augmentation_transform,
    compare_with_without_augmentation
)

from .transfer_learning import (
    load_pretrained_resnet,
    freeze_layers,
    compare_scratch_vs_pretrained,
    visualize_frozen_vs_unfrozen
)

from .visualizations import (
    plot_conv2d_animation,
    plot_feature_maps,
    plot_kernel_gallery,
    plot_pooling_comparison,
    plot_architecture_diagram,
    plot_receptive_field,
    plot_parameter_comparison,
    plot_feature_hierarchy,
    plot_augmentation_grid,
    plot_confusion_matrix_cnn
)

from .widgets import (
    conv2d_widget,
    pooling_widget,
    augmentation_widget,
    architecture_explorer_widget,
    training_widget
)

from .ce_examples import (
    generate_crack_dataset,
    generate_land_use_dataset,
    generate_pavement_distress_dataset,
    generate_vibration_signals,
    make_spectrogram
)

__all__ = [
    # Convolution
    'conv1d', 'conv2d', 'conv2d_multichannel', 'output_size',
    'conv2d_step_by_step', 'common_kernels',
    # Pooling
    'max_pool2d', 'avg_pool2d', 'global_avg_pool', 'pool_step_by_step',
    # Architectures
    'SimpleCNN', 'LeNet5', 'make_vgg_block', 'ResidualBlock', 'SimpleResNet',
    'count_parameters', 'model_summary',
    # Training
    'train_one_epoch', 'evaluate', 'train_cnn', 'plot_training_history',
    'get_predictions',
    # Augmentation
    'augment_numpy', 'augmentation_gallery', 'get_augmentation_transform',
    'compare_with_without_augmentation',
    # Transfer Learning
    'load_pretrained_resnet', 'freeze_layers', 'compare_scratch_vs_pretrained',
    'visualize_frozen_vs_unfrozen',
    # Visualizations
    'plot_conv2d_animation', 'plot_feature_maps', 'plot_kernel_gallery',
    'plot_pooling_comparison', 'plot_architecture_diagram', 'plot_receptive_field',
    'plot_parameter_comparison', 'plot_feature_hierarchy', 'plot_augmentation_grid',
    'plot_confusion_matrix_cnn',
    # Widgets
    'conv2d_widget', 'pooling_widget', 'augmentation_widget',
    'architecture_explorer_widget', 'training_widget',
    # CE Examples
    'generate_crack_dataset', 'generate_land_use_dataset',
    'generate_pavement_distress_dataset', 'generate_vibration_signals',
    'make_spectrogram'
]
