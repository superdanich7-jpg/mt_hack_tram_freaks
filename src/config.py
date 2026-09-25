"""
Центральные конфиги путей, дат и гиперпараметров.
Соответствует ai/CONVENTIONS.md (seed фиксирован, конфиги в src/config.py).
"""
from pathlib import Path

RANDOM_SEED: int = 42

# Длина «свежего» окна для оценки уровня пассажиропотока (в неделях)
RECENT_WEEKS: int = 4

ROUTES: list[int] = [1, 7, 11, 12, 17, 25, 26, 28, 50]
ALL_ROUTES: list[int] = [1, 5, 7, 11, 12, 17, 25, 26, 28, 50]
HOURS: list[int] = list(range(24))

HISTORY_START: str = "2025-01-01"
HISTORY_END: str = "2025-10-31"

VALID_START: str = "2025-10-01"  # начало октябрьской валидации
TRAIN_END: str = "2025-09-30"    # конец train-части для валидации

FORECAST_START: str = "2025-11-01"
FORECAST_END: str = "2025-12-31"

DATA_PARQUET_PATH: Path = Path("data/processed/boardings.parquet")
WEATHER_CSV_PATH: Path = Path("data/external/weather.csv")
HOLIDAYS_CSV_PATH: Path = Path("data/external/holidays.csv")

MODELS_DIR: Path = Path("models")
SUBMISSIONS_DIR: Path = Path("submissions")
REPORTS_DIR: Path = Path("reports")

MODEL_V1_PATH: Path = MODELS_DIR / "lgbm_v1.txt"
MODEL_V2_PATH: Path = MODELS_DIR / "lgbm_v2.txt"
MODEL_FINAL_PATH: Path = MODELS_DIR / "lgbm_final.txt"

BASELINE_SUBMISSION_PATH: Path = Path("test_submission.csv")
FORECAST_CSV_PATH: Path = SUBMISSIONS_DIR / "forecast.csv"
SUBMISSION_CSV_PATH: Path = SUBMISSIONS_DIR / "submission.csv"

# Гиперпараметры LightGBM (v1)
LGBM_PARAMS: dict = {
    "objective": "regression_l1",  # MAE — ближе всего к метрике WAPE
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,  # подобрано абляцией: 31 > 63 > 47 > 23 > 127 > 15
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS: int = 100

# Вес блендинга: prediction = BLEND_ALPHA * LGBM + (1 - BLEND_ALPHA) * robust_baseline
BLEND_ALPHA: float = 0.6
