import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray, prefix: str = "Model") -> dict:
    """Evaluate predicted probabilities against true binary/continuous target using ROC-AUC and LogLoss."""
    # Convert continuous target probabilities to binary labels for ROC-AUC evaluation if needed
    if len(np.unique(y_true)) > 2:
        y_true_binary = (y_true >= 0.5).astype(int)
    else:
        y_true_binary = y_true.astype(int)
        
    y_pred_clipped = np.clip(y_pred_prob, 1e-6, 1.0 - 1e-6)
    
    auc = roc_auc_score(y_true_binary, y_pred_clipped)
    loss = log_loss(y_true_binary, y_pred_clipped)
    
    print(f"[{prefix}] ROC-AUC: {auc:.5f} | LogLoss: {loss:.5f}")
    return {"roc_auc": auc, "log_loss": loss}


def rank_transform(preds: np.ndarray) -> np.ndarray:
    """Transform continuous probability predictions to uniform percentile ranks [0, 1]."""
    ranks = np.argsort(np.argsort(preds))
    return ranks / (len(preds) - 1 + 1e-9)
