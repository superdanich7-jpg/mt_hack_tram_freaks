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

# Гиперпараметры LightGBM, найденные Optuna на валидационном фолде 2025-09.
# Проверены на двух фолдах, которых в подборе не участвовали:
# август 0.87915 → 0.88606 (+0.0069), октябрь 0.90585 → 0.90819 (+0.0023).
# Эффект небольшой, но знак прироста стабилен на обоих фолдах,
# поэтому конфигурация принята как финальная (см. reports/final_model_analysis.md, 3.4).
LGBM_TUNED_PARAMS: dict = {
    **LGBM_PARAMS,
    "num_leaves": 65,
    "learning_rate": 0.021557230126332507,
    "min_child_samples": 72,
    "subsample": 0.7773723679026682,
    "colsample_bytree": 0.7358284803271511,
    "reg_alpha": 6.752807943613342,
    "reg_lambda": 9.727853536370453,
}

EARLY_STOPPING_ROUNDS: int = 100

# Параметры финальной модели. По умолчанию — тюнированные Optuna на фолде
# 2025-09 (см. LGBM_TUNED_PARAMS и раздел 3.4 отчёта).
FINAL_LGBM_PARAMS: dict = LGBM_TUNED_PARAMS

# Вес блендинга: prediction = BLEND_ALPHA * LGBM + (1 - BLEND_ALPHA) * robust_baseline
BLEND_ALPHA: float = 0.6

# ---------------------------------------------------------------------------
# V3: финальная конфигурация (см. reports/final_model_analysis.md)
# ---------------------------------------------------------------------------

# Группы признаков V3, включённые в финальную модель.
#
# cross_route - лучшее среднее по трём фолдам (0.89327 против 0.89230
# у базы), положительный знак прироста на 2 фолдах из 3.
#
# extra_external - дополнительные открытые источники: отопительные
# градусо-дни (из имеющейся погоды Open-Meteo) и макро Brent и USD/RUB
# (Yahoo Finance).
#
# ВАЖНО: световой день (daylight_hours) и темнота УДАЛЕНЫ из этой группы.
# daylight_hours - монотонная функция dayofyear, то есть ровно тот
# сезонный календарный признак, который проект осознанно исключил
# (SEASONAL_EXCLUDE: деревья не умеют экстраполировать за обучающий
# диапазон). На горизонте ноябрь-декабрь световой день выходит НИЖЕ
# минимума обучения (7.00 ч против 7.13 ч) для 21 дня из 61 (34%), и
# модель залипает на январском листе: 31% суммы прогноза получает
# множитель 0.63 от октябрьской нормы вместо 0.91. Подтверждено на
# платформе: 0.87597 -> 0.83831. Группа осталась в коде как
# воспроизводимый эксперимент (`daylight`).
#
# Остальные признаки extra_external проверены на выход за обучающий
# диапазон: 0 из 36 признаков выходят за границы.
# trend - разложение «уровень + рост». Идея верная, но эффект не
# подтвердился: на трёх временных фолдах -0.00125, на помесячном
# бэктесте апрель-октябрь -0.00002. В финальную модель не входит.
FINAL_EXTRA_GROUPS: tuple[str, ...] = ("cross_route", "extra_external")

# Ансамбль НЕ используется в финальной модели: ни simple average,
# ни weighted average, ни стэкинг (в т.ч. с out-of-time весами) не превзошли
# одиночный LightGBM на тестовом месяце (0.90585). Конфигурация сохранена как
# воспроизводимый эксперимент: `python -m src.predict --ensemble`.
FINAL_ENSEMBLE: dict[str, dict] = {
    "LightGBM": {"weight": 1.0, "rounds": 552},
}

# Модель для выгрузки feature_importance (по умолчанию первая в ансамбле)
IMPORTANCE_MODEL: str = "LightGBM"

# Путь к CSV с важностью признаков V3
FEATURE_IMPORTANCE_V3_PATH: Path = MODELS_DIR / "feature_importance_v3.csv"
