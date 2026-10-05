"""Métricas de evaluación para clasificación binaria desbalanceada."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def evaluate(y_true, y_pred, y_score=None) -> dict:
    res = {
        "Recall (Malignant)": recall_score(y_true, y_pred, zero_division=0),
        "Precision (Malignant)": precision_score(y_true, y_pred, zero_division=0),
        "F1 (Malignant)": f1_score(y_true, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Specificity": recall_score(1 - np.asarray(y_true), 1 - np.asarray(y_pred),
                                    zero_division=0),
    }
    if y_score is not None:
        res["ROC-AUC"] = roc_auc_score(y_true, y_score)
        res["PR-AUC"] = average_precision_score(y_true, y_score)
    else:
        res["ROC-AUC"] = np.nan
        res["PR-AUC"] = np.nan
    return res


def results_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(3)
