"""
Модуль построения признаков для прогноза почасового пассажиропотока.

Ключевое правило: исторические агрегаты считаются ТОЛЬКО по строкам
с date < cutoff (история), чтобы исключить утечку из будущего.
Для валидации на октябре cutoff = 2025-10-01, для финального прогноза
cutoff = 2025-11-01.

Соответствует ai/FEATURES.md.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

DATA_PARQUET_PATH: Path = Path("data/processed/boardings.parquet")
WEATHER_CSV_PATH: Path = Path("data/external/weather.csv")
HOLIDAYS_CSV_PATH: Path = Path("data/external/holidays.csv")
DAYLIGHT_CSV_PATH: Path = Path("data/external/daylight.csv")
MACRO_CSV_PATH: Path = Path("data/external/macro.csv")

# База отопительного градусо-дня (стандарт IMD) и окна накопления, часов
HDD_BASE_C: float = 18.0
HDD_WINDOWS: tuple[int, ...] = (24, 72, 168)

# Длина "свежего" окна для оценки тренда уровня (в неделях)
RECENT_WEEKS: int = 4

# Константы для V3-признаков (см. ai/CONVENTIONS.md — никаких магических чисел)
HOURS_IN_DAY: int = 24
DAYS_IN_WEEK: int = 7
PRECIP_WINDOW_6H: int = 6
PRECIP_WINDOW_12H: int = 12
PRECIP_WINDOW_24H: int = 24
PRECIP_THRESHOLD_MM: float = 0.1
WIND_KMH_TO_MS: float = 3.6

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

# ---------------------------------------------------------------------------
# V3: дополнительные группы признаков (включаются точечно через `extra=...`).
# Включение по-умолчанию пустое, чтобы конфигурация v2 осталась воспроизводимой.
# Все новые агрегаты считаются ИСКЛЮЧИТЕЛЬНО по строкам date < cutoff.
# ---------------------------------------------------------------------------
EXTRA_GROUPS: dict[str, list[str]] = {
    # Циклическое кодирование (альтернатива категориальным hour/dow)
    "cyclical": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    # Производные погодные признаки: ощущаемая температура и накопленные осадки
    "weather_derived": [
        "apparent_temperature",
        "precip_6h",
        "precip_12h",
        "precip_24h",
        "snow_24h",
        "is_precipitation",
    ],
    # «Профиль дня»: доля часа в суточном объёте маршрута + ожидаемое значение
    "profile": ["profile_share", "profile_pred", "dow_level_ratio"],
    # Волатильность спроса внутри ячейки и между днями
    "volatility": ["hist_rhd_std", "hist_rhd_cv", "day_cv", "day_cv_recent"],
    # Перекрёстные (cross-route) признаки: активность города целиком
    "cross_route": ["city_hd_med", "route_share_hd", "city_trend_ratio"],
    # Световой день: в декабре на 3,5 часа короче октября
    "daylight": ["daylight_hours", "darkness_hours"],
    # Отопительные градусо-дни: накопленная суровость зимы
    "hdd": ["hdd_24h", "hdd_72h", "hdd_168h"],
    # Макро-ряды: нефть и курс доллара
    "macro": ["brent", "usd_rub"],
    # Все дополнительные внешние источники сразу
    "extra_external": [
        "daylight_hours",
        "darkness_hours",
        "hdd_24h",
        "hdd_72h",
        "hdd_168h",
        "brent",
        "usd_rub",
    ],
}

EXTRA_ALL: tuple[str, ...] = tuple(EXTRA_GROUPS)



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
    extra: tuple[str, ...] = (),
) -> list[str]:
    """
    Возвращает список колонок-признаков для модели.

    :param use_weather: включать погодные признаки.
    :param exclude: признаки, которые нужно исключить (для абляций).
    :param include: дополнительные признаки, которые нужно включить
        (например, погодные при use_weather=False).
    :param extra: подключённые группы признаков V3 (см. EXTRA_GROUPS).
    """
    base = list(FEATURE_COLS) if use_weather else list(FEATURE_COLS_BASE)
    for group in extra:
        if group not in EXTRA_GROUPS:
            raise KeyError(f"Неизвестная группа признаков: {group}")
        base.extend(EXTRA_GROUPS[group])
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

# ---------------------------------------------------------------------------
# V3: дополнительные группы признаков (все — строго по строкам date < cutoff)
# ---------------------------------------------------------------------------

def _merge_keep_keys(df: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Left-merge без конфликта имён ключей (pandas иначе плодит dow_x/dow_y)."""
    renames = {k: f"{k}__key" for k in keys}
    return (
        df.merge(
            right.rename(columns=renames),
            left_on=keys,
            right_on=[renames[k] for k in keys],
            how="left",
        ).drop(columns=list(renames.values()))
    )


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Синус/косинус кодирование суточного и недельного циклов."""
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / HOURS_IN_DAY)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / HOURS_IN_DAY)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / DAYS_IN_WEEK)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / DAYS_IN_WEEK)
    return df


def _add_weather_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    Производные погодные признаки: ощущаемая температура (формула Стедмана)
    и накопленные осадки/снег за 6/12/24 часа.

    Окна скользят «назад» по времени (включая текущий час) — утечки target нет.
    """
    if "temperature_2m" not in df.columns:
        return df

    temp = df["temperature_2m"].astype(float)
    hum = df["relative_humidity_2m"].astype(float).clip(1.0, 100.0)
    wind_kmh = df["wind_speed_10m"].astype(float)

    e_sat = 6.105 * np.exp(17.27 * temp / (237.7 + temp))  # давление насыщения, гПа
    df["apparent_temperature"] = (
        temp + 0.33 * (hum / 100.0) * e_sat - 0.70 * (wind_kmh / WIND_KMH_TO_MS) - 4.00
    )

    acc = df[["route", "date", "hour", "precipitation", "snowfall"]].sort_values(
        ["route", "date", "hour"]
    )
    grp = acc.groupby("route")
    acc["precip_6h"] = grp["precipitation"].transform(
        lambda s: s.rolling(PRECIP_WINDOW_6H, min_periods=1).sum()
    )
    acc["precip_12h"] = grp["precipitation"].transform(
        lambda s: s.rolling(PRECIP_WINDOW_12H, min_periods=1).sum()
    )
    acc["precip_24h"] = grp["precipitation"].transform(
        lambda s: s.rolling(PRECIP_WINDOW_24H, min_periods=1).sum()
    )
    acc["snow_24h"] = grp["snowfall"].transform(
        lambda s: s.rolling(PRECIP_WINDOW_24H, min_periods=1).sum()
    )
    acc = acc.drop(columns=["precipitation", "snowfall"])

    df = df.merge(acc, on=["route", "date", "hour"], how="left")
    for col in ("precip_6h", "precip_12h", "precip_24h", "snow_24h"):
        df[col] = df[col].fillna(0.0)
    df["is_precipitation"] = (df["precip_24h"] > PRECIP_THRESHOLD_MM).astype("int32")
    return df
def _add_profile_features(df: pd.DataFrame, cutoff: pd.Timestamp, recent_weeks: int) -> pd.DataFrame:
    """
    «Профиль дня»: какая доля суточного объёма маршрута приходится на час
    (route x dow x hour), и ожидаемое значение = доля x свежий уровень дня.

    Доля нормируется на суточный итог, поэтому устойчива к общему дрейфу уровня;
    сам дрейф приходит в profile_pred через свежий уровень.
    """
    hist = df[df["date"] < cutoff]
    day_tot = (
        hist.groupby(["route", "date"])["boardings"]
        .sum()
        .rename("day_tot")
        .reset_index()
    )
    hist_p = hist.merge(day_tot, on=["route", "date"], how="left")
    hist_p = hist_p[hist_p["day_tot"] > 0]

    hist_p["share"] = hist_p["boardings"] / hist_p["day_tot"]
    share_rhd = (
        hist_p.groupby(["route", "dow", "hour"])["share"]
        .mean()
        .rename("profile_share")
        .reset_index()
    )

    recent_start = cutoff - pd.Timedelta(weeks=recent_weeks)
    recent = hist[hist["date"] >= recent_start]
    if recent.empty:
        recent = hist
    recent_rd = (
        recent.groupby(["route", "dow"])["boardings"]
        .mean()
        .rename("recent_rd_mean")
        .reset_index()
    )
    recent_r = (
        recent.groupby("route")["boardings"].mean().rename("recent_r_mean").reset_index()
    )

    df = _merge_keep_keys(df, share_rhd, ["route", "dow", "hour"])
    df = _merge_keep_keys(df, recent_rd, ["route", "dow"])
    df = _merge_keep_keys(df, recent_r, ["route"])

    df["profile_share"] = df["profile_share"].fillna(1.0 / (HOURS_IN_DAY * DAYS_IN_WEEK))
    df["profile_pred"] = df["profile_share"] * df["recent_rd_mean"] * HOURS_IN_DAY
    df["dow_level_ratio"] = np.where(
        df["recent_r_mean"] > 0, df["recent_rd_mean"] / df["recent_r_mean"], 1.0
    ).clip(0.2, 5.0)

    return df.drop(columns=["recent_rd_mean", "recent_r_mean"])


def _add_volatility_features(df: pd.DataFrame, cutoff: pd.Timestamp, recent_weeks: int) -> pd.DataFrame:
    """Волатильность спроса внутри ячейки route x hour x dow и между днями."""
    hist = df[df["date"] < cutoff]

    rhd = (
        hist.groupby(["route", "hour", "dow"])["boardings"]
        .agg(hist_rhd_std="std", hist_rhd_mean="mean")
        .reset_index()
    )
    rhd["hist_rhd_cv"] = (rhd["hist_rhd_std"] / (rhd["hist_rhd_mean"] + 1.0)).clip(0, 5.0)
    rhd = rhd.drop(columns=["hist_rhd_mean"])

    day_tot = hist.groupby(["route", "date"])["boardings"].sum().rename("day_tot").reset_index()
    day_cv = (
        day_tot.groupby("route")["day_tot"]
        .apply(lambda s: float(s.std()) / (float(s.mean()) + 1.0))
        .clip(0, 5.0)
        .rename("day_cv")
        .reset_index()
    )
    recent_start = cutoff - pd.Timedelta(weeks=recent_weeks)
    day_cv_recent = (
        day_tot[day_tot["date"] >= recent_start]
        .groupby("route")["day_tot"]
        .apply(lambda s: float(s.std()) / (float(s.mean()) + 1.0))
        .clip(0, 5.0)
        .rename("day_cv_recent")
        .reset_index()
    )

    df = _merge_keep_keys(df, rhd, ["route", "hour", "dow"])
    df = _merge_keep_keys(df, day_cv, ["route"])
    df = _merge_keep_keys(df, day_cv_recent, ["route"])

    for col in ("hist_rhd_std", "hist_rhd_cv", "day_cv", "day_cv_recent"):
        df[col] = df[col].fillna(0.0)
    return df


def _add_cross_route_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, recent_weeks: int
) -> pd.DataFrame:
    """
    Перекрёстные признаки: суммарная активность всех известных маршрутов города.

    city_hd_med — медианный поток по (час, день недели) сразу по всем маршрутам;
    работает как общий «индикатор дня» (праздник, аномальная погода), который
    дерево не может вывести из признаков одного маршрута.
    city_trend_ratio — дрейф уровня города: свежее окно / вся история.
    """
    hist = df[df["date"] < cutoff]
    city_dh = (
        hist.groupby(["date", "hour"])["boardings"].sum().rename("city_boardings").reset_index()
    )
    city_dh["dow"] = city_dh["date"].dt.dayofweek.astype("int32")

    city_hd = (
        city_dh.groupby(["hour", "dow"])["city_boardings"]
        .median()
        .rename("city_hd_med")
        .reset_index()
    )
    city_total_hd = (
        city_dh.groupby(["hour", "dow"])["city_boardings"]
        .mean()
        .rename("city_hd_mean")
        .reset_index()
    )
    route_hd = (
        hist.groupby(["route", "hour", "dow"])["boardings"].mean().rename("route_hd_mean").reset_index()
    )
    share = route_hd.merge(city_total_hd, on=["hour", "dow"], how="left")
    share["route_share_hd"] = (share["route_hd_mean"] / (share["city_hd_mean"] + 1.0)).clip(0, 1.0)
    share = share.drop(columns=["route_hd_mean"])

    recent_start = cutoff - pd.Timedelta(weeks=recent_weeks)
    hist_mean = float(city_dh["city_boardings"].mean())
    recent_mean = float(city_dh[city_dh["date"] >= recent_start]["city_boardings"].mean())
    trend = recent_mean / hist_mean if hist_mean > 0 else 1.0

    df = _merge_keep_keys(df, city_hd, ["hour", "dow"])
    df = _merge_keep_keys(df, share, ["route", "hour", "dow"])
    df["city_hd_med"] = df["city_hd_med"].fillna(0.0)
    df["route_share_hd"] = df["route_share_hd"].fillna(0.0)
    df["city_trend_ratio"] = float(np.clip(trend, 0.3, 3.0))
    return df


def _fill_daily_gap(series: pd.Series, pad_days: int = 10) -> pd.Series:
    """
    Заполняет пропуски дневного ряда по КАЛЕНДАРЮ дат, а не по строкам кадра.

    Это принципиально: заливка по строкам зависит от их порядка, и одни
    и те же данные давали бы разные признаки для одного и того же дня.
    Пропуски возникают на нерабочих днях (1 января, выходные).

    :param pad_days: на сколько дней расширить диапазон влево, чтобы
        закрыть ведущие пропуски (1 января - нерабочий день биржи).
    """
    if series.dropna().empty:
        return series
    first = series.dropna().index.min() - pd.Timedelta(days=pad_days)
    last = series.index.max()
    full = pd.date_range(first, last, freq="D")
    out = series.reindex(full).ffill().bfill()
    # Имя индекса нужно сохранить, иначе reset_index() назовёт колонку "index"
    out.index.name = series.index.name
    return out


def _add_daylight_and_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Световой день (Open-Meteo) и макро-ряды (Brent, USD/RUB из Yahoo Finance).

    Оба файла дневные, поэтому присоединяются по date и размазываются
    на все часы этого дня. Отсутствие файла - не ошибка: функция
    возвращает кадр с пустыми колонками, чтобы форма не ломалась.
    """
    out = df

    if DAYLIGHT_CSV_PATH.exists():
        dl = pd.read_csv(DAYLIGHT_CSV_PATH, parse_dates=["date"]).set_index("date")
        filled = {
            col: _fill_daily_gap(dl[col])
            for col in ("daylight_hours", "darkness_hours")
            if col in dl.columns
        }
        out = _merge_keep_keys(out, pd.DataFrame(filled).reset_index(), ["date"])
    else:
        logging.getLogger(__name__).warning(
            "Файл %s не найден: признаки светового дня будут пустыми. "
            "Запусти `python -m src.external.extra_data`.",
            DAYLIGHT_CSV_PATH,
        )
        out = out.assign(daylight_hours=np.nan, darkness_hours=np.nan)

    if MACRO_CSV_PATH.exists():
        mac = (
            pd.read_csv(MACRO_CSV_PATH, parse_dates=["date"])
            .rename(columns={"BZ=F": "brent", "USDRUB=X": "usd_rub"})
            .set_index("date")
        )
        # Собираем НОВЫЙ фрейм: присваивание серии с расширенным индексом
        # обратно в mac выровнялось бы по индексу и потеряло бы добавленные даты.
        filled = {
            col: _fill_daily_gap(mac[col])
            for col in ("brent", "usd_rub")
            if col in mac.columns
        }
        out = _merge_keep_keys(out, pd.DataFrame(filled).reset_index(), ["date"])
    else:
        logging.getLogger(__name__).warning(
            "Файл %s не найден: признаки нефти и курса будут пустыми. "
            "Запусти `python -m src.external.extra_data`.",
            MACRO_CSV_PATH,
        )
        out = out.assign(brent=np.nan, usd_rub=np.nan)

    return out


def _add_heating_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Отопительные градусо-дни (HDD) из уже имеющейся погоды.

    Сумма max(0, 18 - t) за последние 1/3/7 суток - классический
    индикатор суровости зимы. Лучше мгновенной температуры объясняет
    переход «пешком или на машине»: декабрь холоднее октября, и это
    различие видно не по одному часу, а по накопленной сумме.
    """
    if "temperature_2m" not in df.columns:
        return df
    # Сортируем для скользящих окон, но обязательно возвращаем исходный
    # порядок строк: make_features обязан сохранять построчное соответствие.
    out = df.sort_values(["route", "date", "hour"]).copy()
    out["_hdd"] = (HDD_BASE_C - out["temperature_2m"].astype(float)).clip(lower=0.0)
    for window in HDD_WINDOWS:
        out[f"hdd_{window}h"] = (
            out.groupby("route")["_hdd"]
            .transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        )
    return out.drop(columns=["_hdd"]).reindex(df.index)


def _add_extra_features(
    df: pd.DataFrame, cutoff: pd.Timestamp, extra: tuple[str, ...], recent_weeks: int
) -> pd.DataFrame:
    """Добавление выбранных групп признаков V3."""
    if "cyclical" in extra:
        df = _add_cyclical_features(df)
    if "weather_derived" in extra:
        df = _add_weather_derived(df)
    if "profile" in extra:
        df = _add_profile_features(df, cutoff, recent_weeks)
    if "volatility" in extra:
        df = _add_volatility_features(df, cutoff, recent_weeks)
    if "cross_route" in extra:
        df = _add_cross_route_features(df, cutoff, recent_weeks)
    # Световой день и макро лежат в одном загрузчике: добавляем оба,
    # затем отбрасываем ненужное, чтобы подгруппы работали независимо.
    need_daylight = "daylight" in extra or "extra_external" in extra
    need_macro = "macro" in extra or "extra_external" in extra
    if need_daylight or need_macro:
        df = _add_daylight_and_macro(df)
        if not need_macro:
            df = df.drop(columns=["brent", "usd_rub"], errors="ignore")
        if not need_daylight:
            df = df.drop(
                columns=["daylight_hours", "darkness_hours"], errors="ignore"
            )
    if "hdd" in extra or "extra_external" in extra:
        df = _add_heating_degree_days(df)
    return df






def make_features(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    use_weather: bool = True,
    use_holidays: bool = True,
    recent_weeks: int = RECENT_WEEKS,
    extra: tuple[str, ...] = (),
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
    :param extra: подключаемые группы признаков V3 (см. EXTRA_GROUPS).
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
    out = _add_extra_features(out, cutoff, extra, recent_weeks)
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
