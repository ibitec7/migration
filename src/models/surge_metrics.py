import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def detect_surge(y, threshold_std=1.5, window_mean=None, window_std=None):
    if window_mean is None:
        window_mean = np.mean(y)
    if window_std is None:
        window_std = np.std(y)
    return (y > (window_mean + threshold_std * window_std)).astype(int)

def evaluate_surge_performance(y_true, y_pred, threshold_std=1.5):
    # Detect actual surges
    y_true_mean, y_true_std = np.mean(y_true), np.std(y_true)
    true_surges = detect_surge(y_true, threshold_std, y_true_mean, y_true_std)
    
    # Detect predicted surges (using same historical threshold rules)
    pred_surges = detect_surge(y_pred, threshold_std, y_true_mean, y_true_std)
    
    if np.sum(true_surges) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "surges_found": 0}
        
    return {
        "precision": precision_score(true_surges, pred_surges, zero_division=0),
        "recall": recall_score(true_surges, pred_surges, zero_division=0),
        "f1": f1_score(true_surges, pred_surges, zero_division=0),
        "surges_found": int(np.sum(true_surges))
    }
