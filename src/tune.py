"""
Подбор гиперпараметров бустинга через Optuna.

Режимы:
- `tune_model`   — подбор на одном временном фолде,
- `run_selection` — перебор заранее заданных кандидатов
  (LightGBM/CatBoost/XGBoost x конфигурации) на одном фолде.

Оптимизируем целевую метрику хакатона (WAPE-score = 1 - WAPE) поверх MAE-loss;
раннюю остановку отключаем, чтобы все trial'ы были сопоставимы по числу итераций.
"""
import json
import logging
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from src import config as cfg
from src.metrics import compute_metrics, round_predictions
from src.models_zoo import predict_model, train_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Фиксированное число итераций для честного сравнения trial'ов
N_ROUNDS: int = 1200

# Бюджет Optuna: CatBoost с MAE-loss заметно медленнее, поэтому trial'ов меньше
TRIALS_DEFAULT: dict[str, int] = {"LightGBM": 25, "XGBoost": 20, "CatBoost": 8}

# Диапазоны поиска (не слишком широкие: данных всего ~65 тыс. строк)
SEARCH_SPACE: dict[str, dict[str, tuple[float, float]]] = {
    "LightGBM": {
        "num_leaves": (15, 127),
        "learning_rate": (0.02, 0.15),
        "min_child_samples": (5, 100),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
    },
    "CatBoost": {
        "depth": (4, 10),
        "learning_rate": (0.02, 0.15),
        "l2_leaf_reg": (0.5, 20.0),
        "random_strength": (0.0, 3.0),
    },
    "XGBoost": {
        "max_depth": (3, 12),
        "learning_rate": (0.02, 0.15),
        "min_child_weight": (1, 50),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
    },
}

# Целочисленные параметры (suggest_int), остальные — float
INT_PARAMS: frozenset[str] = frozenset(
    {"num_leaves", "min_child_samples", "depth", "max_depth", "min_child_weight"}
)


def _suggest(trial: optuna.Trial, name: str) -> dict[str, Any]:
    """Пробует значения параметров из SEARCH_SPACE для указанной модели."""
    params: dict[str, Any] = {}
    for key, (low, high) in SEARCH_SPACE[name].items():
        if key == "learning_rate":
            params[key] = trial.suggest_float(key, low, high, log=True)
        elif key in INT_PARAMS:
            params[key] = trial.suggest_int(key, int(low), int(high))
        else:
            params[key] = trial.suggest_float(key, low, high)
    return params


def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    n_trials: int | None = None,
    seed: int = cfg.RANDOM_SEED,
) -> dict[str, Any]:
    """
    Optuna-подбор для одной модели. Целевая функция — WAPE-score на валидации.

    :param n_trials: число trial'ов (по умолчанию из TRIALS_DEFAULT).
    :return: лучшие параметры, WAPE-score и полная история trial'ов.
    """
    n_trials = TRIALS_DEFAULT.get(name, 20) if n_trials is None else n_trials

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, name)
        try:
            model, _ = train_model(
                name, X_train, y_train, X_valid, y_valid, params=params, n_rounds=N_ROUNDS
            )
            preds = round_predictions(predict_model(name, model, X_valid))
        except Exception as exc:  # нестабильные конфигурации просто пропускаем
            logger.debug("Trial %s упал: %s", trial.number, exc)
            return -1.0
        return compute_metrics(y_valid, preds)["WAPE_score"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"v3_{name}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    logger.info(
        "%s: лучший WAPE-score=%.5f за %d trial'ов", name, best.value, len(study.trials)
    )
    return {
        "model": name,
        "best_params": _normalise_params(best.params),
        "best_score": float(best.value),
        "n_trials": len(study.trials),
        "history": [
            {"number": t.number, "score": float(t.value), "params": t.params}
            for t in study.trials
        ],
    }


def _normalise_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Приводит типы параметров к ожидаемым библиотеками.

    Optuna отдаёт целые как int, но при сериализации в JSON они могут
    превратиться в float, а XGBoost затем падает с «max_depth expect int».
    """
    return {
        k: (int(v) if k in INT_PARAMS else float(v)) for k, v in params.items()
    }


def run_selection(dataset_builder, candidates: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Перебор заранее заданных кандидатов на одном фолде.

    :param dataset_builder: функция -> (X_train, y_train, X_valid, y_valid).
    :param candidates: имя модели -> {"label", "params", "n_rounds"}.
    :return: DataFrame с метриками, отсортированный по WAPE-score (по убыванию).
    """
    X_train, y_train, X_valid, y_valid = dataset_builder()
    rows: list[dict[str, Any]] = []
    for name, spec in candidates.items():
        model, rounds = train_model(
            name,
            X_train,
            y_train,
            X_valid,
            y_valid,
            params=spec.get("params"),
            n_rounds=spec.get("n_rounds"),
        )
        preds = round_predictions(predict_model(name, model, X_valid))
        m = compute_metrics(y_valid, preds)
        rows.append(
            {
                "label": spec["label"],
                "model": name,
                "rounds": rounds,
                "WAPE": m["WAPE"],
                "WAPE_score": m["WAPE_score"],
                "MAE": m["MAE"],
            }
        )
        logger.info(
            "%-44s WAPE-score=%.5f MAE=%.2f", spec["label"], m["WAPE_score"], m["MAE"]
        )
    return pd.DataFrame(rows).sort_values("WAPE_score", ascending=False).reset_index(drop=True)


def save_study(result: dict[str, Any], out_path: Path) -> None:
    """Сохраняет результат Optuna в JSON (артефакт для команды)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    print("Модуль запускается через python -m src.experiment_v3 --stage tune")


