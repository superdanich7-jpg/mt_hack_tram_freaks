"""
Метрики качества прогноза. Единая точка расчета WAPE / WAPE-score / MAE.
"""
import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """
    WAPE = Σ|y - ŷ| / Σy
    WAPE-score = max(0, 1 - WAPE)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sum_abs_err = float(np.sum(np.abs(y_true - y_pred)))
    sum_y = float(np.sum(y_true))

    wape = sum_abs_err / sum_y if sum_y > 0 else 1.0
    return {
        "WAPE": float(wape),
        "WAPE_score": float(max(0.0, 1.0 - wape)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "sum_abs_error": sum_abs_err,
        "sum_true": sum_y,
    }


def metrics_frame(metrics_dict: dict) -> pd.DataFrame:
    """Преобразование метрик в однострочный DataFrame для отчётов."""
    return pd.DataFrame([metrics_dict])


def round_predictions(preds) -> np.ndarray:
    """Постобработка прогноза: clip(>=0) и округление до целого."""
    arr = np.asarray(preds, dtype=float)
    arr = np.clip(arr, 0.0, None)
    return np.rint(arr).astype(np.int64)
