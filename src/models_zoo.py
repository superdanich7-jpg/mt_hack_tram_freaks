"""
Зоопарк градиентного бустинга: LightGBM, CatBoost, XGBoost.

Единый интерфейс fit/predict для всех библиотек, чтобы их можно было
подставлять в ансамбль и сравнивать на одинаковых фолдах (см. src/cv.py).

Метрика обучения — L1 (MAE), потому что целевая метрика хакатона WAPE
также является нормированной L1-ошибкой (см. ai/DO_NOT.md: не оптимизировать RMSE).
"""
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src import config as cfg
from src.features import CATEGORICAL_FEATURES

# Базовые параметры, общие для всех деревьев решений
BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "random_state": cfg.RANDOM_SEED,
    "n_jobs": -1,
}

LGBM_PARAMS: dict[str, Any] = {
    **BASE_PARAMS,
    "objective": "regression_l1",
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
}

CATBOOST_PARAMS: dict[str, Any] = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "random_state": cfg.RANDOM_SEED,
    "thread_count": -1,
    "loss_function": "MAE",
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "verbose": 0,
    "allow_writing_files": False,
}

XGB_PARAMS: dict[str, Any] = {
    **BASE_PARAMS,
    "objective": "reg:absoluteerror",
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "verbosity": 0,
}


def _cat_cols(columns) -> list[str]:
    return [c for c in CATEGORICAL_FEATURES if c in set(columns)]


def _cast_for_boost(df: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    """Категориальные колонки -> int-коды (нужно для XGBoost и общих путей)."""
    out = df.copy()
    for col in cats:
        out[col] = out[col].astype("int32")
    return out

def fit_lgbm(X_train, y_train, X_valid, y_valid, params=None, n_rounds=None):
    """LightGBM с early stopping по L1. Возвращает модель и число деревьев."""
    p = dict(LGBM_PARAMS, **(params or {}))
    model = lgb.LGBMRegressor(**p)
    fit_kwargs: dict[str, Any] = {"categorical_feature": _cat_cols(X_train.columns)}
    if n_rounds is not None:
        # Финальный режим: фиксированное число итераций, без early stopping
        model.set_params(n_estimators=n_rounds)
    else:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["eval_metric"] = "l1"
        fit_kwargs["callbacks"] = [lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False)]
    model.fit(X_train, y_train, **fit_kwargs)
    if n_rounds is not None:
        return model, int(n_rounds)
    best = int(getattr(model, "best_iteration_", 0) or p.get("n_estimators", 0))
    return model, max(best, 1)


def fit_catboost(X_train, y_train, X_valid, y_valid, params=None, n_rounds=None):
    """CatBoost с early stopping по MAE."""
    p = dict(CATBOOST_PARAMS, **(params or {}))
    model = CatBoostRegressor(**p)
    fit_kwargs: dict[str, Any] = {"cat_features": _cat_cols(X_train.columns)}
    if n_rounds is not None:
        model.set_params(iterations=n_rounds)
    else:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["early_stopping_rounds"] = cfg.EARLY_STOPPING_ROUNDS
    model.fit(X_train, y_train, **fit_kwargs)
    if n_rounds is not None:
        return model, int(n_rounds)
    best = int(getattr(model, "best_iteration_", 0) or p.get("n_estimators", 0))
    return model, max(best, 1)


def fit_xgb(X_train, y_train, X_valid, y_valid, params=None, n_rounds=None):
    """XGBoost с objective=reg:absoluteerror и ранней остановкой."""
    p = dict(XGB_PARAMS, **(params or {}))
    cats = _cat_cols(X_train.columns)
    model = XGBRegressor(**p)
    fit_kwargs: dict[str, Any] = {"eval_set": [(X_valid, y_valid)]}
    if n_rounds is not None:
        model.set_params(n_estimators=n_rounds)
    else:
        # В XGBoost 3.x ранняя остановка задаётся в конструкторе, а не в fit().
        # Метрику указываем явно (MAE), иначе early stopping считает по RMSE
        # и на objective=reg:absoluteerror останавливается слишком рано —
        # наблюдалось 69 итераций вместо ~500 на коротком обучении.
        model.set_params(
            early_stopping_rounds=cfg.EARLY_STOPPING_ROUNDS,
            eval_metric="mae",
        )
    model.fit(_cast_for_boost(X_train, cats), y_train, **fit_kwargs)
    if n_rounds is not None:
        return model, int(n_rounds)
    best = int(getattr(model, "best_iteration", 0) or p.get("n_estimators", 0))
    return model, max(best, 1)

# Реестр моделей: имя -> обучение и предсказание
MODEL_REGISTRY: dict[str, dict[str, Callable]] = {
    "LightGBM": {
        "fit": fit_lgbm,
        "predict": lambda m, X: m.predict(X),
    },
    "CatBoost": {
        "fit": fit_catboost,
        "predict": lambda m, X: m.predict(X),
    },
    "XGBoost": {
        "fit": fit_xgb,
        "predict": lambda m, X: m.predict(_cast_for_boost(X, _cat_cols(X.columns))),
    },
}


def train_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any] | None = None,
    n_rounds: int | None = None,
) -> tuple[Any, int]:
    """
    Обучает модель из реестра по имени.

    :param name: ключ MODEL_REGISTRY.
    :param n_rounds: если задано, обучение без early stopping (финальная модель).
    :return: обученная модель и число деревьев/итераций.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Неизвестная модель: {name}. Доступны: {list(MODEL_REGISTRY)}")
    model, rounds = MODEL_REGISTRY[name]["fit"](
        X_train, y_train, X_valid, y_valid, params=params, n_rounds=n_rounds
    )
    return model, rounds


def predict_model(name: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    """Предсказание моделью из реестра."""
    return np.asarray(MODEL_REGISTRY[name]["predict"](model, X), dtype=float)



