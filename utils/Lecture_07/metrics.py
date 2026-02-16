"""
Classification Evaluation Metrics

This module provides comprehensive classification metrics including
confusion matrix, accuracy, precision, recall, F1, ROC, PR curves, and AUC.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes=2):
    """
    Compute confusion matrix.
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples,)
        True labels
    y_pred : array_like, shape (n_samples,)
        Predicted labels
    n_classes : int
        Number of classes (default 2 for binary)
    
    Returns:
    --------
    array_like, shape (n_classes, n_classes)
        Confusion matrix where C[i, j] = count of samples
        with true label i predicted as label j
    
    Notes:
    ------
    For binary classification:
        [[TN, FP],
         [FN, TP]]
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Parameters:
    -----------
    y_true : array_like
        True labels
    y_pred : array_like
        Predicted labels
    
    Returns:
    --------
    float
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, pos_label=1):
    """
    Compute precision (positive predictive value).
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_pred : array_like
        Predicted binary labels
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        Precision = TP / (TP + FP)
    
    Notes:
    ------
    Measures: "Of all positive predictions, how many were correct?"
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred, pos_label=1):
    """
    Compute recall (sensitivity, true positive rate).
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_pred : array_like
        Predicted binary labels
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        Recall = TP / (TP + FN)
    
    Notes:
    ------
    Measures: "Of all actual positives, how many did we find?"
    """
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def specificity(y_true, y_pred, pos_label=1):
    """
    Compute specificity (true negative rate).
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_pred : array_like
        Predicted binary labels
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        Specificity = TN / (TN + FP)
    
    Notes:
    ------
    Measures: "Of all actual negatives, how many did we correctly identify?"
    """
    tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)


def f1_score(y_true, y_pred, pos_label=1):
    """
    Compute F1 score (harmonic mean of precision and recall).
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_pred : array_like
        Predicted binary labels
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Notes:
    ------
    Balances precision and recall. Useful for imbalanced datasets.
    """
    prec = precision(y_true, y_pred, pos_label)
    rec = recall(y_true, y_pred, pos_label)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def compute_all_classification_metrics(y_true, y_pred, pos_label=1):
    """
    Compute all classification metrics at once.
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_pred : array_like
        Predicted binary labels
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    dict
        Dictionary with keys: 'Accuracy', 'Precision', 'Recall',
        'Specificity', 'F1', 'TP', 'TN', 'FP', 'FN'
    """
    cm = confusion_matrix(y_true, y_pred, n_classes=2)
    
    # Extract TP, TN, FP, FN
    if pos_label == 1:
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
    else:
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]
    
    return {
        'Accuracy': accuracy(y_true, y_pred),
        'Precision': precision(y_true, y_pred, pos_label),
        'Recall': recall(y_true, y_pred, pos_label),
        'Specificity': specificity(y_true, y_pred, pos_label),
        'F1': f1_score(y_true, y_pred, pos_label),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


def roc_curve(y_true, y_scores, pos_label=1):
    """
    Compute ROC curve (TPR vs FPR at different thresholds).
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples,)
        True binary labels
    y_scores : array_like, shape (n_samples,)
        Predicted probabilities or decision scores
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    dict
        Contains:
        - 'fpr': False positive rates
        - 'tpr': True positive rates
        - 'thresholds': Decision thresholds
    
    Notes:
    ------
    TPR = TP / (TP + FN) = Recall
    FPR = FP / (FP + TN) = 1 - Specificity
    """
    # Get unique thresholds (sorted descending)
    thresholds = np.unique(y_scores)[::-1]
    
    # Add boundary thresholds
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])
    
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        # Compute TPR and FPR
        tp = np.sum((y_true == pos_label) & (y_pred == 1))
        fn = np.sum((y_true == pos_label) & (y_pred == 0))
        fp = np.sum((y_true != pos_label) & (y_pred == 1))
        tn = np.sum((y_true != pos_label) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return {
        'fpr': np.array(fpr_list),
        'tpr': np.array(tpr_list),
        'thresholds': thresholds
    }


def pr_curve(y_true, y_scores, pos_label=1):
    """
    Compute Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array_like, shape (n_samples,)
        True binary labels
    y_scores : array_like, shape (n_samples,)
        Predicted probabilities or decision scores
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    dict
        Contains:
        - 'precision': Precision values
        - 'recall': Recall values
        - 'thresholds': Decision thresholds
    
    Notes:
    ------
    More informative than ROC for imbalanced datasets.
    """
    # Get unique thresholds (sorted descending)
    thresholds = np.unique(y_scores)[::-1]
    
    # Add boundary thresholds
    thresholds = np.concatenate([[np.inf], thresholds, [-np.inf]])
    
    precision_list = []
    recall_list = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        prec = precision(y_true, y_pred, pos_label)
        rec = recall(y_true, y_pred, pos_label)
        
        precision_list.append(prec)
        recall_list.append(rec)
    
    return {
        'precision': np.array(precision_list),
        'recall': np.array(recall_list),
        'thresholds': thresholds
    }


def auc(x, y):
    """
    Compute Area Under Curve using trapezoidal rule.
    
    Parameters:
    -----------
    x : array_like
        X coordinates (must be sorted)
    y : array_like
        Y coordinates
    
    Returns:
    --------
    float
        Area under the curve
    
    Notes:
    ------
    Uses trapezoidal rule: AUC = Σ (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
    """
    # Sort by x
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Trapezoidal rule
    area = 0.0
    for i in range(len(x_sorted) - 1):
        width = x_sorted[i + 1] - x_sorted[i]
        height = (y_sorted[i] + y_sorted[i + 1]) / 2
        area += width * height
    
    return area


def roc_auc_score(y_true, y_scores, pos_label=1):
    """
    Compute ROC AUC score.
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_scores : array_like
        Predicted probabilities or scores
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        ROC AUC score [0, 1]
    
    Notes:
    ------
    AUC = 1.0: Perfect classifier
    AUC = 0.5: Random guessing
    AUC < 0.5: Worse than random
    """
    roc = roc_curve(y_true, y_scores, pos_label)
    return auc(roc['fpr'], roc['tpr'])


def pr_auc_score(y_true, y_scores, pos_label=1):
    """
    Compute Precision-Recall AUC score.
    
    Parameters:
    -----------
    y_true : array_like
        True binary labels
    y_scores : array_like
        Predicted probabilities or scores
    pos_label : int
        Label of positive class (default 1)
    
    Returns:
    --------
    float
        PR AUC score [0, 1]
    """
    pr = pr_curve(y_true, y_scores, pos_label)
    return auc(pr['recall'], pr['precision'])


def plot_confusion_matrix(cm, class_names=None, normalize=False, 
                         cmap='Blues', figsize=(8, 6)):
    """
    Plot confusion matrix as heatmap.
    
    Parameters:
    -----------
    cm : array_like, shape (n_classes, n_classes)
        Confusion matrix
    class_names : list, optional
        Class names for labels
    normalize : bool
        Normalize by row (true labels)
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Annotate cells
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(fpr, tpr, auc_score=None, figsize=(8, 6)):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : array_like
        False positive rates
    tpr : array_like
        True positive rates
    auc_score : float, optional
        AUC score to display
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    label = f'ROC Curve (AUC = {auc_score:.3f})' if auc_score else 'ROC Curve'
    ax.plot(fpr, tpr, linewidth=2, label=label, color='#2E86AB')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig, ax


def plot_pr_curve(recall, precision, auc_score=None, figsize=(8, 6)):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    recall : array_like
        Recall values
    precision : array_like
        Precision values
    auc_score : float, optional
        PR AUC score to display
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot PR curve
    label = f'PR Curve (AUC = {auc_score:.3f})' if auc_score else 'PR Curve'
    ax.plot(recall, precision, linewidth=2, label=label, color='#A23B72')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig, ax
