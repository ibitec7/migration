import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def detect_surge(y, threshold_std=1.5, window_mean=None, window_std=None):
    if window_mean is None:
        window_mean = np.mean(y)
    if window_std is None:
        window_std = np.std(y)
    return (y > (window_mean + threshold_std * window_std)).astype(int)

def evaluate_surge_performance(y_true, y_pred, threshold_std=1.5, train_mean=None, train_std=None):
    """Evaluate surge detection performance against a threshold.

    Args:
        y_true: Ground-truth values from the evaluation period.
        y_pred: Model-predicted values.
        threshold_std: Number of standard deviations above the reference mean that
            defines a surge event.
        train_mean: Mean of the target variable computed exclusively on training
            data.  When provided, the surge threshold is anchored to the training
            distribution, which prevents test-set statistics from leaking into the
            evaluation.  If ``None``, the mean of ``y_true`` is used as a fallback
            (not recommended for production evaluation).
        train_std: Standard deviation of the target variable computed exclusively
            on training data.  See ``train_mean`` for details.  Falls back to the
            std of ``y_true`` when ``None``.

    Returns:
        dict with keys ``precision``, ``recall``, ``f1``, and ``surges_found``.
    """
    # Anchor the surge threshold to the training distribution to avoid leakage.
    ref_mean = train_mean if train_mean is not None else np.mean(y_true)
    ref_std  = train_std  if train_std  is not None else np.std(y_true)

    true_surges = detect_surge(y_true, threshold_std, ref_mean, ref_std)
    pred_surges = detect_surge(y_pred, threshold_std, ref_mean, ref_std)

    if np.sum(true_surges) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "surges_found": 0}

    return {
        "precision": precision_score(true_surges, pred_surges, zero_division=0),
        "recall": recall_score(true_surges, pred_surges, zero_division=0),
        "f1": f1_score(true_surges, pred_surges, zero_division=0),
        "surges_found": int(np.sum(true_surges))
    }
