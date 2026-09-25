"""
Сквозной пайплайн: данные → обучение → прогноз → валидация сабмита.

Использование:
    python -m src.main            # полный цикл
    python -m src.main --skip-eda # без повторных экспериментов v1/v2
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("src.main")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный ML-пайплайн проекта")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Не переобучать v1/v2 (только финальный прогноз)")
    parser.add_argument("--rounds", type=int, default=None, help="Число деревьев финальной модели")
    args = parser.parse_args()

    logger.info("Шаг 1/5: подготовка данных")
    from src.data import load_and_prepare

    df = load_and_prepare()
    logger.info("Данные: %s, маршрутов %d, дней %d", df.shape, df["route"].nunique(),
                df["date"].nunique())

    if not args.skip_experiments:
        logger.info("Шаг 2/5: обучение и валидация моделей v1/v2")
        from src.train import run_training, write_report

        v1 = run_training(use_weather=False, model_path=cfg.MODEL_V1_PATH)
        write_report(v1, False, cfg.REPORTS_DIR / "model_v1.md", "LightGBM v1 (без погоды)")
        logger.info("v1 WAPE-score=%.4f", v1["metrics"]["WAPE_score"])

        v2 = run_training(
            use_weather=True,
            model_path=cfg.MODEL_V2_PATH,
            exclude=("month", "dayofyear", "weekofyear", "day"),
            recent_weeks=8,
        )
        write_report(v2, True, cfg.REPORTS_DIR / "model_v2.md", "LightGBM v2 (+ погода)")
        v2["importance"].to_csv(cfg.MODELS_DIR / "feature_importance.csv", index=False)
        logger.info("v2 WAPE-score=%.4f", v2["metrics"]["WAPE_score"])
    else:
        logger.info("Шаг 2/5: пропущен (--skip-experiments)")

    logger.info("Шаг 3/5: финальный прогноз ноябрь–декабрь")
    from src.predict import apply_route5_zeros, build_submission, train_final_model

    model, future, rounds, cols = train_final_model(rounds=args.rounds)
    model.booster_.save_model(str(cfg.MODEL_FINAL_PATH))

    cfg.SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    forecast = apply_route5_zeros(future)[["route", "date", "hour", "prediction"]].copy()
    forecast["date"] = forecast["date"].dt.strftime("%Y-%m-%d")
    forecast.to_csv(cfg.FORECAST_CSV_PATH, sep=";", index=False, encoding="utf-8")
    submission = build_submission(future)
    submission.to_csv(cfg.SUBMISSION_CSV_PATH, sep=";", index=False, encoding="utf-8")

    logger.info("Шаг 4/5: артефакты модели")
    import pandas as pd

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    importance = (
        pd.Series(model.feature_importances_, index=cols, name="importance")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    importance.to_csv(cfg.MODELS_DIR / "feature_importance.csv", index=False, encoding="utf-8")
    (cfg.MODELS_DIR / "feature_list.json").write_text(
        json.dumps(
            {
                "features": cols,
                "n_features": len(cols),
                "n_rounds": rounds,
                "use_weather": True,
                "recent_weeks": 8,
                "exclude": ["month", "dayofyear", "weekofyear", "day"],
                "train_period": [cfg.HISTORY_START, cfg.HISTORY_END],
                "forecast_period": [cfg.FORECAST_START, cfg.FORECAST_END],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Шаг 5/5: валидация сабмита")
    from src.validate_submission import validate

    ok = validate(cfg.SUBMISSION_CSV_PATH)
    logger.info("Сабмит: %s (строк %d)", "ОК" if ok else "ОШИБКИ", len(submission))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
