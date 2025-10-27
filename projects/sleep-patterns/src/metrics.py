from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

def multiclass_specificity(y_true, y_pred, labels=None) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        denom = tn + fp
        specs.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specs) if len(specs) else 0.0)

def compute_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    return {
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),   # sensibilidad
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity_macro": multiclass_specificity(y_true, y_pred, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }

def to_table_row(name: str, m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": name,
        "precision_macro": m["precision_macro"],
        "recall_macro": m["recall_macro"],
        "f1_macro": m["f1_macro"],
        "specificity_macro": m["specificity_macro"],
    }

def metrics_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False)