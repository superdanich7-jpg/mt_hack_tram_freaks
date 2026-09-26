"""
Финальное обучение на всей истории и генерация сабмита.

Схема:
1) Внутренний сплит (train 2025-01..09 → valid 2025-10) определяет число
   деревьев по early stopping.
2) Финальная модель обучается на всей истории (2025-01-01 … 2025-10-31)
   с найденным числом деревьев.
3) Прогноз по сетке 10 маршрутов × 61 день × 24 часа (14 640 строк),
   route=5 обнуляется, значения clip(≥0) и округляются.

Использование:
    python -m src.predict
"""
import argparse
import logging
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
from src.metrics import round_predictions
from src.train import run_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Финальная конфигурация, выбранная по 2-месячным прокси-валидациям
# (train→+2 мес): без сезонных календарных признаков, с погодой, окно 8 недель.
FINAL_USE_WEATHER: bool = True
FINAL_EXCLUDE: tuple[str, ...] = ("month", "dayofyear", "weekofyear", "day")
FINAL_RECENT_WEEKS: int = 8
ROUNDS_MULTIPLIER: float = 1.1


def build_forecast_grid() -> pd.DataFrame:
    """Полная сетка: 10 маршрутов × 61 день × 24 часа."""
    dates = pd.date_range(cfg.FORECAST_START, cfg.FORECAST_END, freq="D")
    idx = pd.MultiIndex.from_product(
        [cfg.ALL_ROUTES, dates, cfg.HOURS], names=["route", "date", "hour"]
    )
    grid = idx.to_frame(index=False)
    grid["boardings"] = 0.0  # заглушка: агрегаты считаются только по истории
    return grid


def determine_rounds(
    use_weather: bool,
    exclude: tuple[str, ...],
    recent_weeks: int,
    extra: tuple[str, ...] = (),
) -> int:
    """Число деревьев из early stopping на октябрьской валидации."""
    result = run_training(
        use_weather=use_weather,
        model_path=None,
        exclude=exclude,
        valid_start=cfg.VALID_START,
        save=False,
        recent_weeks=recent_weeks,
        extra=extra,
    )
    best = int(result["metrics"]["best_iteration"]) or 500
    rounds = max(50, int(round(best * ROUNDS_MULTIPLIER)))
    logger.info(
        "Early stopping: best_iteration=%d -> финальных деревьев=%d (WAPE-score=%.4f)",
        best,
        rounds,
        result["metrics"]["WAPE_score"],
    )
    return rounds


def train_final_model(
    use_weather: bool = FINAL_USE_WEATHER,
    exclude: tuple[str, ...] = FINAL_EXCLUDE,
    recent_weeks: int = FINAL_RECENT_WEEKS,
    rounds: int | None = None,
    extra: tuple[str, ...] | None = None,
):
    """Обучение финальной модели на всей доступной истории + признаки для прогноза."""
    if use_weather and not weather_available():
        logger.warning(
            "Кэш погоды data/external/weather.csv не найден — прогноз без погодных "
            "признаков. Запусти `python -m src.external.weather`, чтобы их вернуть."
        )
        use_weather = False

    extra = cfg.FINAL_EXTRA_GROUPS if extra is None else extra
    history = load_and_prepare()
    grid = build_forecast_grid()

    if rounds is None:
        rounds = determine_rounds(use_weather, exclude, recent_weeks, extra)

    full = pd.concat([history, grid], ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    feats = make_features(
        full, cutoff=cfg.FORECAST_START, use_weather=use_weather,
        recent_weeks=recent_weeks, extra=extra,
    )
    cols = get_feature_columns(use_weather=use_weather, exclude=exclude, extra=extra)

    train_mask = feats["date"] < cfg.FORECAST_START
    future_mask = feats["date"] >= cfg.FORECAST_START

    params = dict(cfg.LGBM_PARAMS)
    params["n_estimators"] = rounds
    model = lgb.LGBMRegressor(**params)
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in cols]
    model.fit(
        feats.loc[train_mask, cols],
        feats.loc[train_mask, "boardings"],
        categorical_feature=cat_feats,
    )
    logger.info(
        "Финальная модель обучена: строк %d, признаков %d, деревьев %d",
        int(train_mask.sum()),
        len(cols),
        rounds,
    )

    future = feats.loc[future_mask, ["route", "date", "hour", "recent_rhd_med"]].copy()
    future["prediction"] = np.clip(model.predict(feats.loc[future_mask, cols]), 0.0, None)
    future["prediction"] = round_predictions(future["prediction"])
    future["baseline_recent"] = round_predictions(future["recent_rhd_med"])
    future = future.drop(columns=["recent_rhd_med"]).reset_index(drop=True)
    return model, future, rounds, cols


def train_final_ensemble(
    use_weather: bool = FINAL_USE_WEATHER,
    extra: tuple[str, ...] | None = None,
    recent_weeks: int = FINAL_RECENT_WEEKS,
    ensemble: dict[str, dict] | None = None,
    rounds_override: dict[str, int] | None = None,
):
    """
    Обучение финального ансамбля на всей истории (2025-01-01 … 2025-10-31).

    Каждая модель обучается с фиксированным числом итераций (без early
    stopping), взятым из валидации на октябре, и прогнозы объединяются
    взвешенным средним с весами из config.FINAL_ENSEMBLE.

    :param extra: группы признаков V3 (по умолчанию config.FINAL_EXTRA_GROUPS).
    :param rounds_override: переопределение числа итераций по моделям.
    :return: (future, cols, сводка по моделям).
    """
    from src.models_zoo import predict_model, train_model

    if use_weather and not weather_available():
        logger.warning(
            "Кэш погоды data/external/weather.csv не найден — прогноз без погодных "
            "признаков. Запусти `python -m src.external.weather`, чтобы их вернуть."
        )
        use_weather = False

    extra = cfg.FINAL_EXTRA_GROUPS if extra is None else extra
    ensemble = cfg.FINAL_ENSEMBLE if ensemble is None else ensemble
    rounds_override = rounds_override or {}

    history = load_and_prepare()
    grid = build_forecast_grid()
    full = pd.concat([history, grid], ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])

    feats = make_features(
        full,
        cutoff=cfg.FORECAST_START,
        use_weather=use_weather,
        recent_weeks=recent_weeks,
        extra=extra,
    )
    cols = get_feature_columns(
        use_weather=use_weather, exclude=FINAL_EXCLUDE, extra=extra
    )
    train_mask = feats["date"] < cfg.FORECAST_START
    future_mask = feats["date"] >= cfg.FORECAST_START

    X_tr = feats.loc[train_mask, cols]
    y_tr = feats.loc[train_mask, "boardings"]

    future = feats.loc[future_mask, ["route", "date", "hour", "recent_rhd_med"]].copy()
    X_fu = feats.loc[future_mask, cols]

    blended = np.zeros(int(future_mask.sum()), dtype=float)
    weight_sum = 0.0
    summary: dict[str, dict] = {}

    for name, spec in ensemble.items():
        rounds = int(rounds_override.get(name, spec["rounds"]))
        weight = float(spec["weight"])
        model, _ = train_model(name, X_tr, y_tr, X_valid=X_tr, y_valid=y_tr, n_rounds=rounds)
        raw = predict_model(name, model, X_fu)
        blended += weight * raw
        weight_sum += weight
        summary[name] = {"rounds": rounds, "weight": weight, "mean_pred": float(np.mean(raw))}
        logger.info("Ансамбль: %s обучена, %d итераций, вес %.2f", name, rounds, weight)
        if name == cfg.IMPORTANCE_MODEL:
            save_feature_importance(model, cols)

    future["prediction"] = round_predictions(blended / max(weight_sum, 1e-9))
    future["baseline_recent"] = round_predictions(future["recent_rhd_med"])
    future = future.drop(columns=["recent_rhd_med"]).reset_index(drop=True)
    return future, cols, summary


def save_feature_importance(
    model, cols: list[str], out_path: Path | None = None
) -> Path | None:
    """
    Сохраняет важность признаков обученной модели (LightGBM gain).

    :return: путь к файлу или None, если модель не LightGBM.
    """
    out_path = out_path or cfg.FEATURE_IMPORTANCE_V3_PATH
    if not hasattr(model, "feature_importances_"):
        return None
    frame = pd.DataFrame(
        {"feature": list(cols), "importance": np.asarray(model.feature_importances_)}
    )
    frame["share"] = frame["importance"] / frame["importance"].sum()
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Важность признаков сохранена: %s", out_path)
    return out_path


def apply_route5_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """route=5 обучать нельзя — обнуляем прогноз."""
    df = df.copy()
    df.loc[df["route"] == 5, "prediction"] = 0
    return df


def format_date_like_template(template: pd.DataFrame | None, dates: pd.Series) -> pd.Series:
    """
    Формат даты как в реальном test_submission.csv (по факту YYYY-MM-DD),
    а не как написано в README.
    """
    if template is not None:
        sample = template["date"].astype(str).iloc[0]
        fmt = "%d.%m.%Y" if "." in sample else "%Y-%m-%d"
    else:
        fmt = "%Y-%m-%d"
    return dates.dt.strftime(fmt)


def build_submission(future: pd.DataFrame, template_path: Path = cfg.BASELINE_SUBMISSION_PATH):
    """Сборка сабмита в точном формате шаблона и в его порядке строк."""
    df = apply_route5_zeros(future)

    template = None
    if template_path.exists():
        template = pd.read_csv(template_path, sep=";", dtype={"date": str})
        logger.info("Шаблон найден: %s (%d строк)", template_path, len(template))

    if template is not None:
        # Точное совпадение формата: берём строки и даты из шаблона
        out = template.drop(columns=["prediction"]).merge(
            df[["route", "date", "hour", "prediction"]].assign(
                date=format_date_like_template(template, df["date"])
            ),
            on=["route", "date", "hour"],
            how="left",
        )
        missing = int(out["prediction"].isna().sum())
        if missing:
            raise ValueError(f"Не заполнено {missing} строк шаблона — проверь сетку.")
        out["prediction"] = out["prediction"].astype(np.int64)
    else:
        out = df[["route", "date", "hour", "prediction"]].copy()
        out["date"] = format_date_like_template(None, out["date"])
        out["prediction"] = out["prediction"].astype(np.int64)
        out = out.sort_values(["route", "date", "hour"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Финальный прогноз ноябрь–декабрь 2025")
    parser.add_argument("--rounds", type=int, default=None, help="Число деревьев (иначе auto)")
    parser.add_argument("--recent-weeks", type=int, default=FINAL_RECENT_WEEKS)
    parser.add_argument("--no-weather", action="store_true", help="Отключить погодные признаки")
    parser.add_argument("--out", type=str, default=str(cfg.SUBMISSION_CSV_PATH))
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Финальный ансамбль LightGBM+CatBoost+XGBoost вместо одиночного LightGBM",
    )
    args = parser.parse_args()

    use_weather = not args.no_weather
    if args.ensemble:
        future, cols, summary = train_final_ensemble(
            use_weather=use_weather, recent_weeks=args.recent_weeks
        )
        rounds = {k: v["rounds"] for k, v in summary.items()}
    else:
        _, future, rounds, cols = train_final_model(
            use_weather=use_weather,
            exclude=FINAL_EXCLUDE,
            recent_weeks=args.recent_weeks,
            rounds=args.rounds,
        )

    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    forecast = apply_route5_zeros(future)
    forecast_out = forecast[["route", "date", "hour", "prediction"]].copy()
    forecast_out["date"] = forecast_out["date"].dt.strftime("%Y-%m-%d")
    forecast_out.to_csv(cfg.FORECAST_CSV_PATH, sep=";", index=False, encoding="utf-8")

    submission = build_submission(future)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, sep=";", index=False, encoding="utf-8")

    print("=" * 60)
    print(f"Колонки признаков ({len(cols)}): {cols}")
    print(f"Деревьев: {rounds}")
    print(f"Строк в сабмите: {len(submission)}")
    print(f"Сумма прогноза: {int(submission['prediction'].sum()):,}")
    print(f"route=5: max={submission.loc[submission.route == 5, 'prediction'].max()}")
    print(f"Сохранено: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
