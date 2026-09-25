"""
Модуль построения признаков для прогноза почасового пассажиропотока.

Ключевое правило: исторические агрегаты считаются ТОЛЬКО по строкам
с date < cutoff (история), чтобы исключить утечку из будущего.
Для валидации на октябре cutoff = 2025-10-01, для финального прогноза
cutoff = 2025-11-01.

Соответствует ai/FEATURES.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PARQUET_PATH: Path = Path("data/processed/boardings.parquet")
WEATHER_CSV_PATH: Path = Path("data/external/weather.csv")
HOLIDAYS_CSV_PATH: Path = Path("data/external/holidays.csv")

# Длина "свежего" окна для оценки тренда уровня (в неделях)
RECENT_WEEKS: int = 4

WEATHER_COLS: list[str] = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
]

CALENDAR_COLS: list[str] = [
    "route",
    "hour",
    "dow",
    "is_weekend",
    "month",
    "day",
    "dayofyear",
    "weekofyear",
    "is_official_holiday",
    "is_preholiday",
    "days_to_holiday",
    "days_after_holiday",
]

HISTORY_COLS: list[str] = [
    "hist_rh_med",
    "hist_rh_mean",
    "hist_rhd_med",
    "hist_rhd_mean",
    "hist_rd_med",
    "hist_rd_mean",
    "hist_r_mean",
    "hist_r_std",
    "hist_rhe_mean",
    "recent_rhd_med",
    "recent_rh_med",
    "recent_rh_mean",
    "trend_ratio",
]

FEATURE_COLS_BASE: list[str] = CALENDAR_COLS + HISTORY_COLS
FEATURE_COLS: list[str] = FEATURE_COLS_BASE + WEATHER_COLS

# Категориальные признаки для LightGBM
CATEGORICAL_FEATURES: list[str] = ["route", "dow", "hour"]


def load_weather(path: str | Path = WEATHER_CSV_PATH) -> pd.DataFrame:
    """Загрузка кэшированных погодных данных."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла {path}. Сначала выполни `python -m src.external.weather` "
            "или запусти пайплайн без погодных признаков (--no-weather)."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def weather_available(path: str | Path = WEATHER_CSV_PATH) -> bool:
    """Есть ли локальный кэш погоды (файл в .gitignore)."""
    return Path(path).exists()


def load_holidays(path: str | Path = HOLIDAYS_CSV_PATH) -> pd.DataFrame:
    """Загрузка календаря праздников."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_feature_columns(
    use_weather: bool = True,
    exclude: tuple[str, ...] = (),
    include: tuple[str, ...] = (),
) -> list[str]:
    """
    Возвращает список колонок-признаков для модели.

    :param use_weather: включать погодные признаки.
    :param exclude: признаки, которые нужно исключить (для абляций).
    :param include: дополнительные признаки, которые нужно включить
        (например, погодные при use_weather=False).
    """
    base = list(FEATURE_COLS) if use_weather else list(FEATURE_COLS_BASE)
    for col in include:
        if col not in base:
            base.append(col)
    return [c for c in base if c not in set(exclude)]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Календарные признаки."""
    df["dow"] = df["date"].dt.dayofweek.astype("int32")
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int32")
    df["month"] = df["date"].dt.month.astype("int32")
    df["day"] = df["date"].dt.day.astype("int32")
    df["dayofyear"] = df["date"].dt.dayofyear.astype("int32")
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype("int32")
    return df


def _add_external_features(
    df: pd.DataFrame,
    weather: pd.DataFrame | None,
    holidays: pd.DataFrame | None,
) -> pd.DataFrame:
    """Джойн погоды (по date+hour) и праздников (по date)."""
    if weather is not None:
        df = df.merge(weather, on=["date", "hour"], how="left")
    if holidays is not None:
        df = df.merge(holidays, on=["date"], how="left")

    if "is_official_holiday" in df.columns:
        df["is_official_holiday"] = df["is_official_holiday"].fillna(0).astype("int32")
        df["is_preholiday"] = df["is_preholiday"].fillna(0).astype("int32")
        df["days_to_holiday"] = df["days_to_holiday"].fillna(99).astype("int32")
        df["days_after_holiday"] = df["days_after_holiday"].fillna(99).astype("int32")
    return df


def _add_history_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, recent_weeks: int = RECENT_WEEKS
) -> pd.DataFrame:
    """
    Исторические агрегаты по срезам route × hour × dow и т.п.
    Считаются только по строкам с date < cutoff.
    """
    hist = df[df["date"] < cutoff]
    if hist.empty:
        raise ValueError(
            f"История до {cutoff.date()} пуста: невозможно построить агрегаты."
        )

    recent_start = cutoff - pd.Timedelta(weeks=recent_weeks)
    recent = hist[hist["date"] >= recent_start]
    if recent.empty:  # защита на случай очень короткой истории
        recent = hist

    rh = (
        hist.groupby(["route", "hour"])["boardings"]
        .agg(hist_rh_med="median", hist_rh_mean="mean")
        .reset_index()
    )
    rhd = (
        hist.groupby(["route", "hour", "dow"])["boardings"]
        .agg(hist_rhd_med="median", hist_rhd_mean="mean")
        .reset_index()
    )
    rd = (
        hist.groupby(["route", "dow"])["boardings"]
        .agg(hist_rd_med="median", hist_rd_mean="mean")
        .reset_index()
    )
    r = (
        hist.groupby(["route"])["boardings"]
        .agg(hist_r_mean="mean", hist_r_std="std")
        .reset_index()
    )
    rhe = (
        hist.groupby(["route", "hour", "is_weekend"])["boardings"]
        .agg(hist_rhe_mean="mean")
        .reset_index()
    )
    recent_rhd = (
        recent.groupby(["route", "hour", "dow"])["boardings"]
        .agg(recent_rhd_med="median")
        .reset_index()
    )
    recent_rh = (
        recent.groupby(["route", "hour"])["boardings"]
        .agg(recent_rh_med="median", recent_rh_mean="mean")
        .reset_index()
    )

    df = df.merge(rh, on=["route", "hour"], how="left")
    df = df.merge(rhd, on=["route", "hour", "dow"], how="left")
    df = df.merge(rd, on=["route", "dow"], how="left")
    df = df.merge(r, on=["route"], how="left")
    df = df.merge(rhe, on=["route", "hour", "is_weekend"], how="left")
    df = df.merge(recent_rhd, on=["route", "hour", "dow"], how="left")
    df = df.merge(recent_rh, on=["route", "hour"], how="left")

    # Тренд уровня пассажиропотока: свежее окно относительно полной истории
    df["trend_ratio"] = np.where(
        df["hist_rh_mean"] > 0,
        df["recent_rh_mean"] / df["hist_rh_mean"],
        1.0,
    )
    df["trend_ratio"] = df["trend_ratio"].clip(0.3, 3.0)

    fill_cascade = {
        "hist_rhd_med": ["hist_rh_med", "hist_rd_med", "hist_r_mean"],
        "hist_rhd_mean": ["hist_rh_mean", "hist_rd_mean", "hist_r_mean"],
        "hist_rh_med": ["hist_rd_med", "hist_r_mean"],
        "hist_rh_mean": ["hist_rd_mean", "hist_r_mean"],
        "hist_rd_med": ["hist_r_mean"],
        "hist_rd_mean": ["hist_r_mean"],
        "hist_rhe_mean": ["hist_rh_mean", "hist_r_mean"],
        "recent_rhd_med": ["recent_rh_med", "hist_rh_med", "hist_r_mean"],
        "recent_rh_med": ["hist_rh_med", "hist_r_mean"],
        "recent_rh_mean": ["hist_rh_mean", "hist_r_mean"],
    }
    for col, fallbacks in fill_cascade.items():
        for fb in fallbacks:
            if fb in df.columns:
                df[col] = df[col].fillna(df[fb])
        df[col] = df[col].fillna(0.0)

    df["hist_r_mean"] = df["hist_r_mean"].fillna(0.0)
    df["hist_r_std"] = df["hist_r_std"].fillna(0.0)
    return df


def make_features(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    use_weather: bool = True,
    use_holidays: bool = True,
    recent_weeks: int = RECENT_WEEKS,
) -> pd.DataFrame:
    """
    Построение полного набора признаков.

    :param df: DataFrame со столбцами route, date, hour, boardings
        (для будущих периодов boardings можно заполнить заглушкой 0).
    :param cutoff: дата начала прогнозного периода. Все агрегаты считаются
        строго по строкам с date < cutoff.
    :param use_weather: подключать ли погодные признаки.
    :param use_holidays: подключать ли признаки праздников.
    :param recent_weeks: длина «свежего» окна для оценки уровня (недели).
    :return: DataFrame с добавленными признаками (строка к строке).
    """
    cutoff = pd.Timestamp(cutoff)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    out = _add_calendar_features(out)

    weather = load_weather() if use_weather else None
    holidays = load_holidays() if use_holidays else None
    out = _add_external_features(out, weather, holidays)

    out = _add_history_features(out, cutoff, recent_weeks=recent_weeks)
    return out


if __name__ == "__main__":
    from src.data import load_and_prepare

    df = load_and_prepare()
    feats = make_features(df, cutoff="2025-10-01")
    cols = get_feature_columns(use_weather=True)
    X = feats[cols]

    print("=" * 60)
    print(f"Строк: {len(feats)}, признаков: {len(cols)}")
    print(f"Признаки: {cols}")
    n_nan = int(X.isna().sum().sum())
    print(f"NaN в признаках: {n_nan}")
    if n_nan:
        print(X.isna().sum()[X.isna().sum() > 0])
    print(f"Форма X: {X.shape}")
    print("=" * 60)
