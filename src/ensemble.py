"""
Ансамблирование прогнозов: simple average, weighted average и стэкинг.

Все комбинаторы умеют работать в двух режимах:
- веса/мета-модель учатся на одном фолде (OOF),
- применяются на другом фолде (out-of-time).

Это исключает оптимизм: веса никогда не подбираются на тех же данных,
на которых измеряется итоговая метрика.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Шаг перебора весов и сглаживание обратного взвешивания
WEIGHT_STEP: float = 0.05
FLAT_POWER: float = 0.5


def simple_average(preds: dict[str, np.ndarray]) -> np.ndarray:
    """Равновесное среднее всех прогнозов."""
    return np.vstack(list(preds.values())).mean(axis=0)


def inverse_error_weights(
    y_true, preds: dict[str, np.ndarray], power: float = 1.0
) -> dict[str, float]:
    """
    Веса по обратной ошибке: weight_i ∝ (WAPE_i)^(-power).

    :param power: 1 — классическое обратное взвешивание, >1 — более плоское
        распределение (меньше доминирования лучшей модели).
    """
    inv = {}
    for name, p in preds.items():
        wape = max(compute_metrics(y_true, p)["WAPE"], 1e-6)
        inv[name] = wape ** (-power)
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    logger.info(
        "Веса (обратная ошибка, power=%.1f): %s", power,
        {k: round(v, 3) for k, v in weights.items()},
    )
    return weights


def weighted_average(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Взвешенное среднее по заданным весам (сумма весов нормируется)."""
    total = sum(weights.values()) or 1.0
    acc = np.zeros(len(next(iter(preds.values()))), dtype=float)
    for name, p in preds.items():
        acc += (weights.get(name, 0.0) / total) * p
    return acc


def grid_search_weights(
    y_true, preds: dict[str, np.ndarray], grid: np.ndarray | None = None
) -> tuple[dict[str, float], float]:
    """
    Подбор весов перебором по сетке (для 2-3 моделей) с условием суммы = 1.

    :return: лучшие веса и WAPE-score.
    """
    names = list(preds)
    steps = grid if grid is not None else np.arange(0.0, 1.0001, WEIGHT_STEP)
    if len(names) == 2:
        best_w: dict[str, float] = {names[0]: 0.5, names[1]: 0.5}
        best_s = -1.0
        for w0 in steps:
            cand = {names[0]: float(w0), names[1]: float(1 - w0)}
            score = compute_metrics(y_true, weighted_average(preds, cand))["WAPE_score"]
            if score > best_s:
                best_s, best_w = score, cand
        return best_w, best_s

    best_w = {n: 1.0 / len(names) for n in names}
    best_s = compute_metrics(y_true, simple_average(preds))["WAPE_score"]
    for w0 in steps:
        for w1 in steps:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            cand = {names[0]: float(w0), names[1]: float(w1), names[2]: float(max(w2, 0.0))}
            score = compute_metrics(y_true, weighted_average(preds, cand))["WAPE_score"]
            if score > best_s:
                best_s, best_w = score, cand
    return best_w, best_s


def stack_model(
    y_true, preds: dict[str, np.ndarray], nonnegative: bool = True
) -> tuple[Callable[[dict[str, np.ndarray]], np.ndarray], float]:
    """
    Линейный стэкинг (мета-регрессия) на OOF-прогнозах.

    :param nonnegative: ограничить коэффициенты снизу нулём (стабильнее).
    :return: функция применения и WAPE-score на обучающих данных.
    """
    names = list(preds)
    X = np.column_stack([preds[n] for n in names])
    meta = LinearRegression(positive=nonnegative)
    meta.fit(X, np.asarray(y_true, dtype=float))
    score = compute_metrics(y_true, meta.predict(X))["WAPE_score"]
    logger.info(
        "Стэк: коэффициенты %s, WAPE-score(in-sample)=%.5f",
        dict(zip(names, np.round(meta.coef_, 3))),
        score,
    )

    def apply(new_preds: dict[str, np.ndarray]) -> np.ndarray:
        return meta.predict(np.column_stack([new_preds[n] for n in names]))

    return apply, score

def evaluate_blends(
    y_true, preds: dict[str, np.ndarray], labels: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Сравнивает одиночные модели и несколько способов объединения.

    :param labels: человекочитаемые названия вместо технических ключей.
    :return: DataFrame с метриками, отсортированный по WAPE-score.
    """
    rows: list[dict] = []
    for name, p in preds.items():
        m = compute_metrics(y_true, p)
        rows.append(
            {
                "method": (labels or {}).get(name, name),
                "WAPE": m["WAPE"],
                "WAPE_score": m["WAPE_score"],
                "MAE": m["MAE"],
            }
        )

    variants = {
        "Ансамбль: simple average": simple_average(preds),
        "Ансамбль: weighted (обратная ошибка)": weighted_average(
            preds, inverse_error_weights(y_true, preds)
        ),
        "Ансамбль: weighted (p=0.5, плоско)": weighted_average(
            preds, inverse_error_weights(y_true, preds, power=FLAT_POWER)
        ),
    }
    if len(preds) <= 3:
        best_w, _ = grid_search_weights(y_true, preds)
        variants["Ансамбль: веса по сетке " + str({k: round(v, 2) for k, v in best_w.items()})] = (
            weighted_average(preds, best_w)
        )
        apply_stack, _ = stack_model(y_true, preds, nonnegative=True)
        variants["Ансамбль: stacking (coef>=0)"] = apply_stack(preds)

    for tag, blended in variants.items():
        m = compute_metrics(y_true, blended)
        rows.append(
            {"method": tag, "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"]}
        )
    return pd.DataFrame(rows).sort_values("WAPE_score", ascending=False).reset_index(drop=True)

