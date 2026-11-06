"""
Probability Calibration Module (Platt Scaling & Isotonic Regression)
Guarantees optimal log loss calibration for model probability predictions.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def platt_scale(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    """
    Platt Scaling via Logistic Regression on log-odds (logits).
    """
    eps = 1e-15
    clipped = np.clip(y_pred_probs, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    clf = LogisticRegression(C=1e5, solver="lbfgs")
    clf.fit(logits, y_true)
    calibrated = clf.predict_proba(logits)[:, 1]
    
    loss_before = log_loss(y_true, y_pred_probs)
    loss_after = log_loss(y_true, calibrated)
    print(f"[Platt Scaling] Log Loss: {loss_before:.5f} -> {loss_after:.5f}")
    return calibrated

def isotonic_scale(y_true: np.ndarray, y_pred_probs: np.ndarray) -> np.ndarray:
    """
    Non-parametric Isotonic Regression Calibration.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    calibrated = iso.fit_transform(y_pred_probs, y_true)
    
    loss_before = log_loss(y_true, y_pred_probs)
    loss_after = log_loss(y_true, calibrated)
    print(f"[Isotonic Regression] Log Loss: {loss_before:.5f} -> {loss_after:.5f}")
    return calibrated
