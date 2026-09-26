"""
Обучение и валидация модели LightGBM.

Схема валидации (см. ai/ML_TASK.md):
- train: 2025-01-01 … 2025-09-30
- valid: 2025-10-01 … 2025-10-31

Использование:
    python -m src.train            # модель v1 (без погоды)
    python -m src.train --weather  # модель v2 (с погодой)
"""
import argparse
import json
import logging
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config as cfg
from src.data import load_and_prepare
from src.features import (
    CATEGORICAL_FEATURES,
    get_feature_columns,
    make_features,
    weather_available,
)
from src.metrics import compute_metrics, round_predictions

# LightGBM 4.7 ругается на аргумент eval_set (deprecated) — предупреждение не по делу
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_datasets(
    use_weather: bool = False,
    exclude: tuple[str, ...] = (),
    valid_start: str = cfg.VALID_START,
    valid_end: str | None = None,
    recent_weeks: int = cfg.RECENT_WEEKS,
):
    """Готовит train/valid матрицы признаков без утечки из будущего."""
    if use_weather and not weather_available():
        logger.warning(
            "Кэш погоды data/external/weather.csv не найден — обучение без погодных "
            "признаков. Запусти `python -m src.external.weather`, чтобы их вернуть."
        )
        use_weather = False

    df = load_and_prepare()
    feats = make_features(
        df, cutoff=valid_start, use_weather=use_weather, recent_weeks=recent_weeks
    )
    cols = get_feature_columns(use_weather=use_weather, exclude=exclude)

    train_mask = feats["date"] < valid_start
    if valid_end is not None:
        valid_mask = (feats["date"] >= valid_start) & (feats["date"] <= valid_end)
    else:
        valid_mask = feats["date"] >= valid_start

    X_train = feats.loc[train_mask, cols]
    y_train = feats.loc[train_mask, "boardings"]
    X_valid = feats.loc[valid_mask, cols]
    y_valid = feats.loc[valid_mask, "boardings"]

    logger.info("Train: %s | Valid: %s | Признаков: %d", X_train.shape, X_valid.shape, len(cols))
    return feats, X_train, y_train, X_valid, y_valid, cols


def train_lgbm(X_train, y_train, X_valid, y_valid, params: dict | None = None):
    """Обучение LightGBM с early stopping по L1 на валидации."""
    params = dict(cfg.LGBM_PARAMS if params is None else params)
    model = lgb.LGBMRegressor(**params)
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        categorical_feature=cat_feats,
    )
    return model


def run_training(
    use_weather: bool = False,
    model_path: Path | None = None,
    exclude: tuple[str, ...] = (),
    params: dict | None = None,
    valid_start: str = cfg.VALID_START,
    valid_end: str | None = None,
    save: bool = True,
    recent_weeks: int = cfg.RECENT_WEEKS,
) -> dict:
    """Полный цикл: подготовка -> обучение -> метрики -> сохранение модели."""
    feats, X_train, y_train, X_valid, y_valid, cols = build_datasets(
        use_weather=use_weather,
        exclude=exclude,
        valid_start=valid_start,
        valid_end=valid_end,
        recent_weeks=recent_weeks,
    )
    model = train_lgbm(X_train, y_train, X_valid, y_valid, params=params)

    raw_pred = model.predict(X_valid)
    pred_round = round_predictions(raw_pred)
    metrics = compute_metrics(y_valid.values, pred_round)

    # Baseline-опора для блендинга: медиана (route, hour, dow) за последние 4 недели
    feats["dow"] = feats["date"].dt.dayofweek.astype("int32")
    if valid_end is not None:
        valid_frame = feats[(feats["date"] >= valid_start) & (feats["date"] <= valid_end)].copy()
    else:
        valid_frame = feats[feats["date"] >= valid_start].copy()
    valid_frame["lgbm_pred"] = np.clip(raw_pred, 0.0, None)
    blend_pred = round_predictions(
        cfg.BLEND_ALPHA * valid_frame["lgbm_pred"].values
        + (1 - cfg.BLEND_ALPHA) * valid_frame["recent_rhd_med"].values
    )
    blend_metrics = compute_metrics(y_valid.values, blend_pred)

    metrics["best_iteration"] = int(getattr(model, "best_iteration_", 0) or 0)
    metrics["use_weather"] = use_weather
    metrics["blend_WAPE_score"] = blend_metrics["WAPE_score"]

    if model_path is not None and save:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(model_path))
        logger.info("Модель сохранена: %s", model_path)

    importance = (
        pd.Series(model.feature_importances_, index=cols, name="importance")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return {
        "metrics": metrics,
        "blend_metrics": blend_metrics,
        "importance": importance,
        "cols": cols,
        "valid_pred": valid_frame["lgbm_pred"].values,
        "y_valid": y_valid.values,
        "valid_frame": valid_frame,
        "valid_start": valid_start,
        "valid_end": str(valid_frame["date"].max().date()),
        "recent_weeks": recent_weeks,
        "model_params": model.get_params(),
        "exclude": list(exclude),
    }


def run_ensemble(
    valid_start: str = cfg.VALID_START,
    weights: tuple[float, ...] = (0.5, 0.5),
    recent_weeks: int = cfg.RECENT_WEEKS,
    params: dict | None = None,
) -> dict:
    """
    Ансамбль двух конфигураций:
      A) полный набор признаков (v1, без погоды);
      B) без сезонных календарных признаков + погода (v2).
    Возвращает метрики каждой модели и их среднего.
    """
    configs = [
        {"name": "A_full", "use_weather": False, "exclude": ()},
        {"name": "B_noseasonal_weather", "use_weather": True,
         "exclude": ("month", "dayofyear", "weekofyear", "day")},
    ]
    preds, y_valid, feats_ref = [], None, None
    per_model = {}
    for c in configs:
        result = run_training(
            use_weather=c["use_weather"],
            model_path=None,
            exclude=c["exclude"],
            valid_start=valid_start,
            save=False,
            recent_weeks=recent_weeks,
            params=params,
        )
        per_model[c["name"]] = result["metrics"]
        preds.append(result["valid_pred"])
        y_valid = result["y_valid"]
        feats_ref = result["valid_frame"]

    ens = np.zeros_like(preds[0], dtype=float)
    for w, p in zip(weights, preds):
        ens += w * p
    ens_round = round_predictions(ens / sum(weights))
    ens_metrics = compute_metrics(y_valid, ens_round)

    # блендинг лучшей конфигурации с медианной опорой
    blend_pred = round_predictions(
        cfg.BLEND_ALPHA * preds[1] + (1 - cfg.BLEND_ALPHA) * feats_ref["recent_rhd_med"].values
    )
    blend_metrics = compute_metrics(y_valid, blend_pred)

    return {
        "per_model": per_model,
        "ensemble": ens_metrics,
        "blend_v2_median": blend_metrics,
        "valid_start": valid_start,
        "recent_weeks": recent_weeks,
    }


def write_report(
    result: dict, use_weather: bool, report_path: Path, model_name: str
) -> None:
    """Формирование markdown-отчёта по модели."""
    m = result["metrics"]
    bm = result["blend_metrics"]
    top = result["importance"].head(20)

    lines = [
        f"# Отчёт по модели {model_name}",
        "",
        "## 1. Постановка эксперимента",
        f"- Train: {cfg.HISTORY_START} … {pd.Timestamp(result['valid_start']).date() - pd.Timedelta(days=1)}",
        f"- Valid: {result['valid_start']} … {result['valid_end']}",
        f"- Модель: LightGBM LGBMRegressor (objective=`{cfg.LGBM_PARAMS['objective']}`)",
        f"- Внешние признаки (погода): {'да' if use_weather else 'нет'}",
        f"- Исключённые признаки: {result.get('exclude') or '—'}",
        f"- Свежее окно уровня: {result['recent_weeks']} нед.",
        f"- Число признаков: {len(result['cols'])}",
        f"- Best iteration (early stopping): {m['best_iteration']}",
        "",
        "## 2. Гиперпараметры",
        "| Параметр | Значение |",
        "|---|---|",
    ]
    for k, v in result["model_params"].items():
        lines.append(f"| `{k}` | {v} |")

    lines += [
        "",
        "## 3. Метрики на октябрьской валидации",
        "| Вариант | WAPE | WAPE-score | MAE |",
        "|---|---|---|---|",
        f"| LightGBM (+постобработка) | {m['WAPE']:.4f} | **{m['WAPE_score']:.4f}** | {m['MAE']:.2f} |",
        f"| Блендинг LGBM + медиана(4 нед) | {bm['WAPE']:.4f} | **{bm['WAPE_score']:.4f}** | {bm['MAE']:.2f} |",
        "",
        "## 4. Топ-20 признаков по важности",
        "| # | Признак | Важность (split) |",
        "|---|---|---|",
    ]
    for i, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(f"| {i} | `{row.feature}` | {int(row.importance)} |")

    lines += [
        "",
        "## 5. Вывод",
        f"- WAPE-score модели: **{m['WAPE_score']:.4f}**, блендинга: **{bm['WAPE_score']:.4f}**.",
        f"- Порог baseline (~0.48) и целевой порог 0.85 {'достигнуты' if bm['WAPE_score'] >= 0.85 else 'не достигнуты'}.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Отчёт записан: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение LightGBM на history 2025-01..2025-10")
    parser.add_argument("--weather", action="store_true", help="Использовать погодные признаки (v2)")
    parser.add_argument("--out", type=str, default=None, help="Путь для сохранения модели")
    parser.add_argument("--report", type=str, default=None, help="Путь для markdown-отчёта")
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Список исключаемых признаков через запятую (для абляций)",
    )
    parser.add_argument("--leaves", type=int, default=None, help="Переопределить num_leaves")
    parser.add_argument("--lr", type=float, default=None, help="Переопределить learning_rate")
    parser.add_argument(
        "--importance-csv",
        type=str,
        default=None,
        help="Путь для сохранения CSV с важностью признаков",
    )
    parser.add_argument(
        "--valid-start",
        type=str,
        default=cfg.VALID_START,
        help="Начало валидационного периода (для прокси-теста на 2 месяца: 2025-09-01)",
    )
    parser.add_argument(
        "--recent-weeks",
        type=int,
        default=cfg.RECENT_WEEKS,
        help="Длина свежего окна для оценки уровня (недели)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Оценить ансамбль двух конфигураций и выйти",
    )
    args = parser.parse_args()

    if args.ensemble:
        res = run_ensemble(
            valid_start=args.valid_start, recent_weeks=args.recent_weeks, params=params
        )
        print("=" * 60)
        for name, m in res["per_model"].items():
            print(f"{name:26s}: WAPE-score={m['WAPE_score']:.4f} (MAE={m['MAE']:.2f})")
        print(f"{'ENSEMBLE (0.5/0.5)':26s}: WAPE-score={res['ensemble']['WAPE_score']:.4f} "
              f"(MAE={res['ensemble']['MAE']:.2f})")
        print(f"{'BLEND v2+median':26s}: WAPE-score={res['blend_v2_median']['WAPE_score']:.4f} "
              f"(MAE={res['blend_v2_median']['MAE']:.2f})")
        print(f"valid_start={res['valid_start']} recent_weeks={res['recent_weeks']}")
        print("=" * 60)
        cache = cfg.REPORTS_DIR / f"ensemble_{args.valid_start}.json"
        cache.write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
        return

    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    params = dict(cfg.LGBM_PARAMS)
    if args.leaves is not None:
        params["num_leaves"] = args.leaves
    if args.lr is not None:
        params["learning_rate"] = args.lr

    use_weather = args.weather
    default_model = cfg.MODEL_V1_PATH if not use_weather else cfg.MODEL_V2_PATH
    model_path = Path(args.out) if args.out else default_model
    report_path = Path(args.report) if args.report else cfg.REPORTS_DIR / (
        "model_v2.md" if use_weather else "model_v1.md"
    )
    model_name = "LightGBM v2 (+ погода)" if use_weather else "LightGBM v1 (без погоды)"

    result = run_training(
        use_weather=use_weather,
        model_path=model_path,
        exclude=exclude,
        params=params,
        valid_start=args.valid_start,
        save=args.out is None,
        recent_weeks=args.recent_weeks,
    )
    write_report(result, use_weather, report_path, model_name)

    if args.importance_csv:
        imp_path = Path(args.importance_csv)
        imp_path.parent.mkdir(parents=True, exist_ok=True)
        result["importance"].to_csv(imp_path, index=False, encoding="utf-8")
        logger.info("Важность признаков записана: %s", imp_path)

    m = result["metrics"]
    print("=" * 60)
    print(f"{model_name}: WAPE={m['WAPE']:.4f} | WAPE-score={m['WAPE_score']:.4f} | MAE={m['MAE']:.2f}")
    print(f"Блендинг:          WAPE-score={result['blend_metrics']['WAPE_score']:.4f}")
    print("=" * 60)

    # Кэш метрик для финального решения
    cache = cfg.REPORTS_DIR / f"metrics_{'v2' if use_weather else 'v1'}.json"
    cache.write_text(json.dumps({"metrics": m, "blend_metrics": result["blend_metrics"]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
