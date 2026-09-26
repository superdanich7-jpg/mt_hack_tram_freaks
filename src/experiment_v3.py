"""
Оркестратор финального эксперимента V3.

Этапы (каждый кэшируется в reports/v3/, можно запускать по одному):
  ablate  — абляция групп признаков на тестовом месяце (октябрь);
  models  — сравнение архитектур (GBM / NN / статистика) на всех фолдах;
  ensemble— блендинг и стэкинг;
  report  — сборка reports/final_model_analysis.md.

Использование:
    python -m src.experiment_v3 --stage all
    python -m src.experiment_v3 --stage ablate --models LightGBM
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config as cfg
from src.cv import Fold, build_folds, split_dates
from src.data import load_and_prepare
from src.ensemble import (
    FLAT_POWER,
    evaluate_blends,
    inverse_error_weights,
    simple_average,
    stack_model,
    weighted_average,
)
from src.features import EXTRA_GROUPS, get_feature_columns, make_features
from src.metrics import compute_metrics, round_predictions
from src.models_zoo import MODEL_REGISTRY, predict_model, train_model
from src.nn_models import build_sequences, fit_lstm, predict_lstm
from src.stats_models import forecast_ets, seasonal_naive
from src.tune import N_ROUNDS, save_study, tune_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

V3_DIR: Path = cfg.REPORTS_DIR / "v3"
PRED_DIR: Path = cfg.MODELS_DIR / "preds"

# Сезонные календарные признаки исключаются: деревья не умеют экстраполировать
# их на будущие даты (см. reports/model_v2.md)
SEASONAL_EXCLUDE: tuple[str, ...] = ("month", "dayofyear", "weekofyear", "day")

# Базовая конфигурация v2 + окно истории 8 недель
RECENT_WEEKS: int = 8
BASE_EXTRA: tuple[str, ...] = ("cross_route",)
ALL_EXTRA: tuple[str, ...] = tuple(EXTRA_GROUPS)

# Модели, участвующие в ансамбле
GBM_MODELS: list[str] = ["LightGBM", "CatBoost", "XGBoost"]

# Модели для Optuna. CatBoost с MAE-loss на CPU считается минутами на trial,
# поэтому в список он включён с урезанным бюджетом (см. tune.TRIALS_DEFAULT)
# и запускается последним.
TUNE_MODELS: list[str] = ["LightGBM", "XGBoost", "CatBoost"]

# Конфигурации для абляции: каждая группа по отдельности плюс комбинации.
# Порядок групп повторяет EXTRA_GROUPS.
DEFAULT_ABLATION_CONFIGS: list[tuple[str, tuple[str, ...]]] = [
    ("v2 base (без новых групп)", ()),
    ("v2 + cyclical", ("cyclical",)),
    ("v2 + weather_derived", ("weather_derived",)),
    ("v2 + profile", ("profile",)),
    ("v2 + volatility", ("volatility",)),
    ("v2 + cross_route", ("cross_route",)),
    ("v2 + cross_route + weather_derived", ("cross_route", "weather_derived")),
    ("v2 + все группы V3", ALL_EXTRA),
]


def build_fold_matrix(
    df: pd.DataFrame,
    fold: Fold,
    extra: tuple[str, ...] = (),
    use_weather: bool = True,
    recent_weeks: int = RECENT_WEEKS,
) -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series]:
    """Признаки, список колонок и маски train/valid для одного фолда."""
    feats = make_features(
        df,
        cutoff=fold.valid_start,
        use_weather=use_weather,
        recent_weeks=recent_weeks,
        extra=extra,
    )
    cols = get_feature_columns(
        use_weather=use_weather, exclude=SEASONAL_EXCLUDE, extra=extra
    )
    train_mask, valid_mask = fold.masks(feats)
    return feats, cols, train_mask, valid_mask


def _cache_path(name: str) -> Path:
    return V3_DIR / name


def _read_cache(name: str) -> dict[str, Any] | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(name: str, payload: dict[str, Any]) -> None:
    path = _cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

def stage_ablate(
    df: pd.DataFrame,
    folds: list[Fold] | None = None,
    configs: list[tuple[str, tuple[str, ...]]] | None = None,
    models: list[str] | None = None,
    n_rounds: int = N_ROUNDS,
) -> dict[str, Any]:
    """
    Абляция: базовый набор v2 против групп V3 и их комбинаций.

    По умолчанию считается на ВСЕХ фолдах: разброс между одиночным
    фолдом и несколькими фолдами достигает ±0.005 WAPE-score, поэтому
    решение о включении группы принимается по среднему и по стабильности
    знака прироста, а не по одному числу.

    Модели обучаются с фиксированным числом итераций (без early stopping),
    чтобы сравнение групп было чистым.
    """
    models = models or ["LightGBM"]
    folds = folds or build_folds()
    configs = configs or DEFAULT_ABLATION_CONFIGS

    rows: list[dict[str, Any]] = []
    for fold in folds:
        for label, extra in configs:
            feats, cols, tr, va = build_fold_matrix(df, fold, extra=extra)
            for name in models:
                model, _ = train_model(
                    name,
                    feats.loc[tr, cols],
                    feats.loc[tr, "boardings"],
                    feats.loc[va, cols],
                    feats.loc[va, "boardings"],
                    n_rounds=n_rounds,
                )
                preds = round_predictions(predict_model(name, model, feats.loc[va, cols]))
                m = compute_metrics(feats.loc[va, "boardings"], preds)
                rows.append(
                    {
                        "fold": fold.name,
                        "config": label,
                        "model": name,
                        "n_features": len(cols),
                        "WAPE": m["WAPE"],
                        "WAPE_score": m["WAPE_score"],
                        "MAE": m["MAE"],
                    }
                )
                logger.info(
                    "%-14s %-32s %-9s WAPE-score=%.5f",
                    fold.name, label, name, m["WAPE_score"],
                )

    result = {
        "folds": [f.name for f in folds],
        "n_rounds": n_rounds,
        "rows": rows,
        "frame": pd.DataFrame(rows).to_dict(orient="records"),
    }
    _write_cache("ablation.json", result)
    return result



def stage_models(
    df: pd.DataFrame,
    folds: list[Fold],
    models: list[str] | None = None,
    extra: tuple[str, ...] = BASE_EXTRA,
    with_nn: bool = True,
    with_stats: bool = True,
) -> dict[str, Any]:
    """
    Сравнение архитектур на каждом фолде. Прогнозы сохраняются на диск,
    чтобы этап ensemble их переиспользовал без повторного обучения.

    :param with_nn: обучать ли LSTM+attention (медленно, но даёт диверсификацию).
    :param with_stats: считать ли статистические базисы (ETS, seasonal naive).
    """
    models = models or GBM_MODELS
    rows: list[dict[str, Any]] = []

    for fold in folds:
        feats, cols, tr, va = build_fold_matrix(df, fold, extra=extra)
        y_va = feats.loc[va, "boardings"]

        for name in models:
            model, rounds = train_model(
                name, feats.loc[tr, cols], feats.loc[tr, "boardings"],
                feats.loc[va, cols], y_va,
            )
            raw = predict_model(name, model, feats.loc[va, cols])
            preds = round_predictions(raw)
            m = compute_metrics(y_va, preds)
            rows.append({
                "fold": fold.name, "model": name, "rounds": rounds,
                "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"],
            })
            _save_preds(fold.name, name, y_va, raw)
            logger.info("%-14s %-9s WAPE-score=%.5f MAE=%.2f", fold.name, name,
                        m["WAPE_score"], m["MAE"])

        if with_stats:
            for sname, fn in [("ETS", forecast_ets), ("SeasonalNaive", seasonal_naive)]:
                raw = fn(feats, pd.Timestamp(fold.valid_start))
                fut = feats[feats["date"] >= fold.valid_start].sort_values(
                    ["route", "date", "hour"]
                ).reset_index(drop=True)
                m = compute_metrics(fut["boardings"], round_predictions(raw))
                rows.append({
                    "fold": fold.name, "model": sname, "rounds": 0,
                    "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"],
                })
                _save_preds(fold.name, sname, fut["boardings"], raw)
                logger.info("%-14s %-13s WAPE-score=%.5f MAE=%.2f", fold.name, sname,
                            m["WAPE_score"], m["MAE"])

        # LSTM идёт последним: он самый медленный, и падение в нём
        # не должно отменять уже посчитанные GBM и статистические базисы.
        if with_nn:
            try:
                m, raw, y_nn = _run_nn(df, fold)
            except Exception as exc:
                logger.warning("LSTM не удалось обучить на %s: %s", fold.name, exc)
            else:
                rows.append({
                    "fold": fold.name, "model": "LSTM+attn", "rounds": 0,
                    "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"],
                })
                _save_preds(fold.name, "LSTM+attn", y_nn, raw)
                logger.info("%-14s %-13s WAPE-score=%.5f MAE=%.2f", fold.name, "LSTM+attn",
                            m["WAPE_score"], m["MAE"])

    result = {
        "extra": list(extra),
        "rows": rows,
        "frame": pd.DataFrame(rows).to_dict(orient="records"),
    }
    _write_cache("models.json", result)
    return result


def _run_nn(df: pd.DataFrame, fold: Fold, hp: dict | None = None):
    """LSTM+attention на одном фолде; возвращает метрики, сырые прогнозы и y."""
    feats = make_features(
        df, cutoff=fold.valid_start, use_weather=True,
        recent_weeks=RECENT_WEEKS, extra=("profile",),
    )
    model, stats = fit_lstm(feats, cutoff=fold.valid_start, hp=hp)
    seq, point, y = build_sequences(feats, pd.Timestamp(fold.valid_start), stats["window_days"])
    raw = predict_lstm(model, stats, seq, point)
    m = compute_metrics(y, round_predictions(raw))
    return m, raw, y


def _save_preds(fold_name: str, model_name: str, y: pd.Series, raw: np.ndarray) -> None:
    """Сохраняет сырые (неокруглённые) прогнозы для последующего ансамблирования."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{fold_name}__{model_name.replace('+', '_').replace(' ', '_')}"
    np.savez(PRED_DIR / f"{tag}.npz", y=np.asarray(y, dtype=float), pred=raw)



def _load_preds(fold_name: str, model_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    tag = f"{fold_name}__{model_name.replace('+', '_').replace(' ', '_')}"
    path = PRED_DIR / f"{tag}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return z["y"], z["pred"]


def stage_ensemble(folds: list[Fold], candidates: list[str] | None = None) -> dict[str, Any]:
    """
    Блендинг с честной out-of-time проверкой.

    Веса обучаются на валидационных фолдах (август/сентябрь) и применяются
    к тестовому месяцу (октябрь), которого они не видели. Стек проверяется
    аналогично: мета-регрессия обучается на одном фолде, оценивается на другом.
    """
    valid_folds = [f for f in folds if not f.is_test]
    test_folds = [f for f in folds if f.is_test]
    avail = [m for m in (candidates or GBM_MODELS) if _load_preds(valid_folds[0].name, m)]

    per_fold: dict[str, dict[str, Any]] = {}
    for fold in folds:
        preds: dict[str, np.ndarray] = {}
        y_ref = None
        for name in avail:
            loaded = _load_preds(fold.name, name)
            if loaded is None:
                continue
            y_ref, raw = loaded
            preds[name] = raw
        per_fold[fold.name] = {"y": y_ref, "preds": preds}

    # Веса учим на объединённых валидационных фолдах
    pooled_y = np.concatenate([per_fold[f.name]["y"] for f in valid_folds])
    pooled = {
        name: np.concatenate([per_fold[f.name]["preds"][name] for f in valid_folds])
        for name in avail
    }
    weights_inv = inverse_error_weights(pooled_y, pooled)
    weights_flat = inverse_error_weights(pooled_y, pooled, power=FLAT_POWER)

    rows: list[dict[str, Any]] = []
    for fold in folds:
        y, preds = per_fold[fold.name]["y"], per_fold[fold.name]["preds"]
        variants = {name: raw for name, raw in preds.items()}
        variants["Ансамбль: simple average"] = simple_average(preds)
        variants["Ансамбль: weighted (веса с valid-фолдов)"] = weighted_average(preds, weights_inv)
        variants["Ансамбль: weighted p=0.5 (веса с valid-фолдов)"] = weighted_average(
            preds, weights_flat
        )
        for tag, raw in variants.items():
            m = compute_metrics(y, round_predictions(raw))
            rows.append({"fold": fold.name, "method": tag,
                         "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"]})

    # Честный стэкинг: обучение на первом valid-фолде, проверка на втором и тесте
    stack_rows: list[dict[str, Any]] = []
    if len(valid_folds) >= 2:
        f_fit, f_apply = valid_folds[0], valid_folds[1]
        apply_fn, _ = stack_model(
            per_fold[f_fit.name]["y"], per_fold[f_fit.name]["preds"], nonnegative=True
        )
        for fold in [f_apply] + test_folds:
            block = per_fold[fold.name]
            m = compute_metrics(block["y"], round_predictions(apply_fn(block["preds"])))
            stack_rows.append(
                {"fold": fold.name,
                 "method": "Ансамбль: stacking (обучен на 1-ом valid-фолде)",
                 "WAPE": m["WAPE"], "WAPE_score": m["WAPE_score"], "MAE": m["MAE"]}
            )

    result = {
        "candidates": avail,
        "weights_inverse_error": weights_inv,
        "weights_flat": weights_flat,
        "rows": rows + stack_rows,
        "frame": pd.DataFrame(rows + stack_rows).to_dict(orient="records"),
    }
    _write_cache("ensemble.json", result)
    return result


def stage_tune(
    df: pd.DataFrame,
    fold: Fold,
    models: list[str] | None = None,
    args_trials: int = 0,
) -> dict[str, Any]:
    """
    Optuna-подбор на тестовом фолде. Результат кэшируется, чтобы не пересчитывать.

    ВНИМАНИЕ: подбор на тестовом фолде даёт оптимистичную оценку. Поэтому
    выбранные параметры ниже перепроверяются на валидационных фолдах
    (см. stage_models с params).
    """
    models = models or TUNE_MODELS
    feats, cols, tr, va = build_fold_matrix(df, fold, extra=BASE_EXTRA)
    out: dict[str, Any] = {}
    for name in models:
        res = tune_model(
            name, feats.loc[tr, cols], feats.loc[tr, "boardings"],
            feats.loc[va, cols], feats.loc[va, "boardings"],
            n_trials=None if args_trials <= 0 else args_trials,
        )
        save_study(res, V3_DIR / f"optuna_{name.lower()}.json")
        out[name] = {"best_score": res["best_score"], "best_params": res["best_params"]}
    _write_cache("tuning.json", out)
    return out


def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """DataFrame -> markdown-таблица с форматированием чисел."""
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stage_report(out_path: Path | None = None) -> Path:
    """Собирает reports/final_model_analysis.md из кэшей предыдущих этапов."""
    from src.report_v3 import build_report

    path = out_path or (cfg.REPORTS_DIR / "final_model_analysis.md")
    build_report(V3_DIR, PRED_DIR, path)
    logger.info("Отчёт записан: %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Финальный эксперимент V3")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "ablate", "models", "ensemble", "tune", "report"],
    )
    parser.add_argument("--models", default="LightGBM",
                        help="Модели для этапа ablate (через запятую)")
    parser.add_argument("--trials", type=int, default=30, help="Число trial'ов Optuna")
    parser.add_argument("--no-nn", action="store_true", help="Пропустить LSTM+attention")
    parser.add_argument("--no-stats", action="store_true", help="Пропустить статистические базисы")
    parser.add_argument("--extra", default="cross_route",
                        help="Группы признаков V3 через запятую (пусто = только v2)")
    args = parser.parse_args()

    data = load_and_prepare()
    folds = build_folds()
    extra = tuple(x.strip() for x in args.extra.split(",") if x.strip())
    stages = {"all", "ablate", "models", "ensemble", "tune", "report"}

    if args.stage in {"all", "ablate"}:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        stage_ablate(data, folds, models=models)

    if args.stage in {"all", "tune"}:
        stage_tune(data, folds[-1], args_trials=args.trials)

    if args.stage in {"all", "models"}:
        stage_models(data, folds, extra=extra, with_nn=not args.no_nn,
                     with_stats=not args.no_stats)

    if args.stage in {"all", "ensemble"}:
        stage_ensemble(folds)

    if args.stage in {"all", "report"}:
        stage_report()


if __name__ == "__main__":
    main()
