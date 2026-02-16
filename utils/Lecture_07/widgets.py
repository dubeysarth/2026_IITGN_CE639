"""
Interactive Widgets for Classification Exploration

This module provides ipywidgets for exploring logistic regression,
decision boundaries, class imbalance, confusion matrices, ROC curves,
and multi-class classification.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import ipywidgets as widgets
    from IPython.display import display
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Widgets will not work.")


def logistic_regression_widget():
    """
    Interactive widget for exploring logistic regression parameters.
    
    Allows adjustment of:
    - Learning rate
    - Number of iterations
    - Regularization strength
    - Dataset noise
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .logistic_regression import logistic_regression_gd, logistic_predict
    from .decision_boundaries import plot_decision_boundary_2d
    from .ce_examples import generate_linearly_separable_data
    
    def update(learning_rate=0.1, n_iterations=500, lambda_reg=0.0, noise=0.3):
        # Generate data
        data = generate_linearly_separable_data(n_samples=200, n_features=2, 
                                               class_sep=2.0, random_state=42)
        X = data['X']
        y = data['y']
        
        # Add noise
        X += np.random.RandomState(42).normal(0, noise, X.shape)
        
        # Train model
        result = logistic_regression_gd(X, y, learning_rate=learning_rate,
                                       n_iterations=n_iterations,
                                       lambda_reg=lambda_reg, verbose=False)
        
        # Create prediction function
        def predict_fn(X_test):
            return logistic_predict(X_test, result['weights'])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Decision boundary
        from .decision_boundaries import create_mesh_grid
        xx, yy = create_mesh_grid(X, resolution=0.02)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_fn(grid_points).reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5, linestyles='--')
        ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='#2E86AB', s=60, 
                   edgecolors='black', linewidths=1, alpha=0.8, label='Class 0')
        ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='#A23B72', s=60, 
                   edgecolors='black', linewidths=1, alpha=0.8, label='Class 1')
        ax1.set_xlabel('Feature 1', fontsize=11)
        ax1.set_ylabel('Feature 2', fontsize=11)
        ax1.set_title(f'Decision Boundary (Acc: {result["accuracy_history"][-1]:.3f})', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Training history
        ax2.plot(result['loss_history'], linewidth=2, color='#2E86AB', label='Loss')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(result['accuracy_history'], linewidth=2, color='#A23B72', 
                     linestyle='--', label='Accuracy')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11, color='#2E86AB')
        ax2_twin.set_ylabel('Accuracy', fontsize=11, color='#A23B72')
        ax2.set_title('Training History', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_twin.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    # Create widgets
    learning_rate_slider = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01,
                                               description='Learning Rate:', style={'description_width': 'initial'})
    iterations_slider = widgets.IntSlider(value=500, min=100, max=2000, step=100,
                                         description='Iterations:', style={'description_width': 'initial'})
    lambda_slider = widgets.FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1,
                                       description='Regularization (λ):', style={'description_width': 'initial'})
    noise_slider = widgets.FloatSlider(value=0.3, min=0.0, max=1.0, step=0.05,
                                      description='Noise:', style={'description_width': 'initial'})
    
    widgets.interact(update, learning_rate=learning_rate_slider, 
                    n_iterations=iterations_slider,
                    lambda_reg=lambda_slider, noise=noise_slider)


def decision_boundary_widget():
    """
    Interactive widget for exploring decision boundary thresholds.
    
    Allows adjustment of:
    - Decision threshold
    - Class separation
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .logistic_regression import logistic_regression_gd, logistic_predict
    from .metrics import compute_all_classification_metrics
    from .ce_examples import generate_linearly_separable_data
    
    # Generate and train once
    data = generate_linearly_separable_data(n_samples=200, n_features=2, 
                                           class_sep=2.0, random_state=42)
    X = data['X']
    y = data['y']
    
    result = logistic_regression_gd(X, y, learning_rate=0.1, n_iterations=500, verbose=False)
    
    def update(threshold=0.5):
        # Predict probabilities
        y_prob = logistic_predict(X, result['weights'])
        y_pred = (y_prob >= threshold).astype(int)
        
        # Compute metrics
        metrics = compute_all_classification_metrics(y, y_pred)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Decision boundary with threshold
        from .decision_boundaries import create_mesh_grid
        xx, yy = create_mesh_grid(X, resolution=0.02)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = logistic_predict(grid_points, result['weights']).reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
        ax1.contour(xx, yy, Z, levels=[threshold], colors='black', linewidths=3, 
                   linestyles='--', label=f'Threshold={threshold:.2f}')
        
        # Plot points colored by prediction
        correct = (y_pred == y)
        ax1.scatter(X[correct & (y == 0), 0], X[correct & (y == 0), 1], 
                   c='#2E86AB', s=60, edgecolors='black', linewidths=1, 
                   alpha=0.8, marker='o')
        ax1.scatter(X[correct & (y == 1), 0], X[correct & (y == 1), 1], 
                   c='#A23B72', s=60, edgecolors='black', linewidths=1, 
                   alpha=0.8, marker='o')
        ax1.scatter(X[~correct, 0], X[~correct, 1], 
                   c='red', s=80, edgecolors='black', linewidths=2, 
                   alpha=0.9, marker='x', label='Misclassified')
        
        ax1.set_xlabel('Feature 1', fontsize=11)
        ax1.set_ylabel('Feature 2', fontsize=11)
        ax1.set_title(f'Decision Boundary (Threshold={threshold:.2f})', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Metrics bar chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [metrics[m] for m in metric_names]
        
        bars = ax2.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#06A77D'],
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Classification Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    threshold_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                          description='Threshold:', style={'description_width': 'initial'})
    
    widgets.interact(update, threshold=threshold_slider)


def class_imbalance_widget():
    """
    Interactive widget for exploring class imbalance handling techniques.
    
    Allows adjustment of:
    - Imbalance ratio
    - Resampling method (None, Oversample, Undersample, SMOTE)
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .class_imbalance import (generate_imbalanced_data, random_oversample, 
                                 random_undersample, smote)
    from .logistic_regression import logistic_regression_gd, logistic_predict
    from .metrics import compute_all_classification_metrics
    
    def update(imbalance_ratio=0.1, method='None'):
        # Generate imbalanced data
        data = generate_imbalanced_data(n_samples=500, imbalance_ratio=imbalance_ratio,
                                       n_features=2, class_sep=2.0, random_state=42)
        X = data['X']
        y = data['y']
        
        # Apply resampling
        if method == 'Oversample':
            X_resampled, y_resampled = random_oversample(X, y, target_ratio=1.0, random_state=42)
        elif method == 'Undersample':
            X_resampled, y_resampled = random_undersample(X, y, target_ratio=1.0, random_state=42)
        elif method == 'SMOTE':
            X_resampled, y_resampled = smote(X, y, k_neighbors=5, target_ratio=1.0, random_state=42)
        else:
            X_resampled, y_resampled = X, y
        
        # Train models
        result_original = logistic_regression_gd(X, y, learning_rate=0.1, 
                                                n_iterations=500, verbose=False)
        result_resampled = logistic_regression_gd(X_resampled, y_resampled, 
                                                 learning_rate=0.1, n_iterations=500, verbose=False)
        
        # Evaluate on original test set
        y_pred_original = (logistic_predict(X, result_original['weights']) >= 0.5).astype(int)
        y_pred_resampled = (logistic_predict(X, result_resampled['weights']) >= 0.5).astype(int)
        
        metrics_original = compute_all_classification_metrics(y, y_pred_original)
        metrics_resampled = compute_all_classification_metrics(y, y_pred_resampled)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original data distribution
        unique, counts = np.unique(y, return_counts=True)
        axes[0, 0].bar(unique, counts, color=['#2E86AB', '#A23B72'], 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_xlabel('Class', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title(f'Original Data (Ratio={imbalance_ratio:.2f})', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(unique)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Resampled data distribution
        unique_r, counts_r = np.unique(y_resampled, return_counts=True)
        axes[0, 1].bar(unique_r, counts_r, color=['#2E86AB', '#A23B72'], 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[0, 1].set_xlabel('Class', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title(f'After {method} Resampling', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(unique_r)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Metrics comparison
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        x = np.arange(len(metric_names))
        width = 0.35
        
        vals_original = [metrics_original[m] for m in metric_names]
        vals_resampled = [metrics_resampled[m] for m in metric_names]
        
        axes[1, 0].bar(x - width/2, vals_original, width, label='Original', 
                      color='#2E86AB', edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[1, 0].bar(x + width/2, vals_resampled, width, label=method, 
                      color='#A23B72', edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[1, 0].set_ylabel('Score', fontsize=11)
        axes[1, 0].set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metric_names)
        axes[1, 0].set_ylim([0, 1.1])
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Confusion matrices
        from .metrics import confusion_matrix
        cm_original = confusion_matrix(y, y_pred_original)
        cm_resampled = confusion_matrix(y, y_pred_resampled)
        
        # Show improvement in recall (minority class detection)
        recall_improvement = metrics_resampled['Recall'] - metrics_original['Recall']
        
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.7, f'Recall Improvement:', ha='center', va='center',
                       fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.5, f'{recall_improvement:+.3f}', ha='center', va='center',
                       fontsize=24, fontweight='bold', 
                       color='green' if recall_improvement > 0 else 'red',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.3, f'(Minority Class Detection)', ha='center', va='center',
                       fontsize=11, transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    imbalance_slider = widgets.FloatSlider(value=0.1, min=0.05, max=0.5, step=0.05,
                                          description='Imbalance Ratio:', 
                                          style={'description_width': 'initial'})
    method_dropdown = widgets.Dropdown(options=['None', 'Oversample', 'Undersample', 'SMOTE'],
                                      value='None', description='Resampling Method:',
                                      style={'description_width': 'initial'})
    
    widgets.interact(update, imbalance_ratio=imbalance_slider, method=method_dropdown)


def confusion_matrix_widget():
    """
    Interactive widget for exploring confusion matrix with threshold adjustment.
    
    Allows adjustment of:
    - Decision threshold
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .logistic_regression import logistic_regression_gd, logistic_predict
    from .metrics import confusion_matrix, plot_confusion_matrix, compute_all_classification_metrics
    from .ce_examples import generate_linearly_separable_data
    
    # Generate and train once
    data = generate_linearly_separable_data(n_samples=300, n_features=2, 
                                           class_sep=2.0, random_state=42)
    X = data['X']
    y = data['y']
    
    result = logistic_regression_gd(X, y, learning_rate=0.1, n_iterations=500, verbose=False)
    y_prob = logistic_predict(X, result['weights'])
    
    def update(threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        
        # Compute confusion matrix and metrics
        cm = confusion_matrix(y, y_pred)
        metrics = compute_all_classification_metrics(y, y_pred)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confusion matrix heatmap
        im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax1)
        
        ax1.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Pred 0', 'Pred 1'],
               yticklabels=['True 0', 'True 1'],
               ylabel='True Label',
               xlabel='Predicted Label',
               title=f'Confusion Matrix (Threshold={threshold:.2f})')
        
        # Annotate cells
        thresh_val = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                label_text = ['TN', 'FP', 'FN', 'TP'][i*2 + j]
                ax1.text(j, i, f'{label_text}\n{cm[i, j]}',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh_val else "black",
                        fontsize=14, fontweight='bold')
        
        # Metrics table
        ax2.axis('off')
        
        table_data = [
            ['Metric', 'Value', 'Formula'],
            ['Accuracy', f'{metrics["Accuracy"]:.3f}', '(TP+TN)/(TP+TN+FP+FN)'],
            ['Precision', f'{metrics["Precision"]:.3f}', 'TP/(TP+FP)'],
            ['Recall', f'{metrics["Recall"]:.3f}', 'TP/(TP+FN)'],
            ['Specificity', f'{metrics["Specificity"]:.3f}', 'TN/(TN+FP)'],
            ['F1 Score', f'{metrics["F1"]:.3f}', '2·Prec·Rec/(Prec+Rec)'],
        ]
        
        table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                         colWidths=[0.25, 0.2, 0.55])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Classification Metrics', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    threshold_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                          description='Threshold:', 
                                          style={'description_width': 'initial'})
    
    widgets.interact(update, threshold=threshold_slider)


def roc_curve_widget():
    """
    Interactive widget for exploring ROC curve with threshold marker.
    
    Allows adjustment of:
    - Decision threshold (shows corresponding point on ROC curve)
    - Dataset imbalance
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .logistic_regression import logistic_regression_gd, logistic_predict
    from .metrics import roc_curve, pr_curve, roc_auc_score, pr_auc_score
    from .class_imbalance import generate_imbalanced_data
    
    def update(threshold=0.5, imbalance_ratio=0.3):
        # Generate data
        data = generate_imbalanced_data(n_samples=500, imbalance_ratio=imbalance_ratio,
                                       n_features=2, class_sep=2.0, random_state=42)
        X = data['X']
        y = data['y']
        
        # Train model
        result = logistic_regression_gd(X, y, learning_rate=0.1, n_iterations=500, verbose=False)
        y_prob = logistic_predict(X, result['weights'])
        
        # Compute ROC and PR curves
        roc = roc_curve(y, y_prob)
        pr = pr_curve(y, y_prob)
        roc_auc = roc_auc_score(y, y_prob)
        pr_auc = pr_auc_score(y, y_prob)
        
        # Find point on ROC curve for current threshold
        y_pred = (y_prob >= threshold).astype(int)
        from .metrics import recall, specificity
        tpr_current = recall(y, y_pred)
        fpr_current = 1 - specificity(y, y_pred)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC curve
        ax1.plot(roc['fpr'], roc['tpr'], linewidth=2, color='#2E86AB', 
                label=f'ROC (AUC={roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        
        # Mark current threshold
        ax1.plot(fpr_current, tpr_current, 'ro', markersize=12, 
                label=f'Threshold={threshold:.2f}')
        
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # PR curve
        ax2.plot(pr['recall'], pr['precision'], linewidth=2, color='#A23B72', 
                label=f'PR (AUC={pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y) / len(y)
        ax2.axhline(baseline, color='k', linestyle='--', linewidth=1.5, 
                   alpha=0.5, label=f'Random (y={baseline:.2f})')
        
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.show()
    
    threshold_slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.05,
                                          description='Threshold:', 
                                          style={'description_width': 'initial'})
    imbalance_slider = widgets.FloatSlider(value=0.3, min=0.05, max=0.5, step=0.05,
                                          description='Imbalance Ratio:', 
                                          style={'description_width': 'initial'})
    
    widgets.interact(update, threshold=threshold_slider, imbalance_ratio=imbalance_slider)


def multiclass_widget():
    """
    Interactive widget for exploring multi-class classification with softmax.
    
    Allows adjustment of:
    - Number of classes
    - Learning rate
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Please install: pip install ipywidgets")
        return
    
    from .logistic_regression import softmax_regression_gd, softmax_predict_class
    from .decision_boundaries import plot_multiclass_boundaries
    
    def update(n_classes=3, learning_rate=0.1):
        # Generate multi-class data
        np.random.seed(42)
        n_per_class = 100
        X_list = []
        y_list = []
        
        for class_idx in range(n_classes):
            angle = 2 * np.pi * class_idx / n_classes
            center = 3 * np.array([np.cos(angle), np.sin(angle)])
            X_class = np.random.randn(n_per_class, 2) * 0.8 + center
            X_list.append(X_class)
            y_list.append(np.full(n_per_class, class_idx))
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # Train model
        result = softmax_regression_gd(X, y, n_classes=n_classes, 
                                      learning_rate=learning_rate,
                                      n_iterations=1000, verbose=False)
        
        # Create prediction function
        def predict_fn(X_test):
            from .logistic_regression import softmax_predict_class
            return softmax_predict_class(X_test, result['weights'])
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Decision boundaries
        from .decision_boundaries import create_mesh_grid
        xx, yy = create_mesh_grid(X, resolution=0.05)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_fn(grid_points).reshape(xx.shape)
        
        # Colors
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        ax1.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=np.arange(n_classes + 1) - 0.5)
        
        for class_idx in range(n_classes):
            mask = (y == class_idx)
            ax1.scatter(X[mask, 0], X[mask, 1], c=[colors[class_idx]], 
                       s=60, edgecolors='black', linewidths=1, 
                       label=f'Class {class_idx}', alpha=0.8)
        
        ax1.set_xlabel('Feature 1', fontsize=11)
        ax1.set_ylabel('Feature 2', fontsize=11)
        ax1.set_title(f'Multi-Class Decision Boundaries (K={n_classes})', 
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Training history
        ax2.plot(result['loss_history'], linewidth=2, color='#2E86AB', label='Loss')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(result['accuracy_history'], linewidth=2, color='#A23B72', 
                     linestyle='--', label='Accuracy')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11, color='#2E86AB')
        ax2_twin.set_ylabel('Accuracy', fontsize=11, color='#A23B72')
        ax2.set_title('Training History', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=9)
        ax2_twin.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final Accuracy: {result['accuracy_history'][-1]:.3f}")
    
    n_classes_slider = widgets.IntSlider(value=3, min=2, max=5, step=1,
                                        description='Number of Classes:', 
                                        style={'description_width': 'initial'})
    learning_rate_slider = widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01,
                                              description='Learning Rate:', 
                                              style={'description_width': 'initial'})
    
    widgets.interact(update, n_classes=n_classes_slider, learning_rate=learning_rate_slider)
