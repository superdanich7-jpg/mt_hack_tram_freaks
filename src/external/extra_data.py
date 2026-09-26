"""
Дополнительные открытые источники для прогноза пассажиропотока.

Каждый источник кэшируется в data/external/ и подключается опционально:
если файла нет, пайплайн продолжает работать без него.

Критерий пригодности источника - жёсткий: данные должны покрывать
И обучение (2025-01-01 … 2025-10-31), И прогноз (2025-11-01 … 2025-12-31).
Иначе признак либо бесполезен (константа), либо создаёт утечку.

Источники:
  1. Световой день (восход/заход) - Open-Meteo Archive, daily.
     В декабре день на ~3,5 часа короче октября; темнота меняет
     мобильность, а это уже за пределами обучающих дат по часам.
  2. Нефть Brent - Yahoo Finance (BZ=F), daily, валюта/сырьё как
     прокси макроэкономической активности.
  3. Курс USD/RUB - Yahoo Finance (USDRUB=X), daily.
  4. Отопительный градусо-дни (HDD) - считается из уже имеющейся
     погоды, отдельная загрузка не нужна.

Использование:
    python -m src.external.extra_data
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EXTERNAL_DIR: Path = Path("data/external")
DAYLIGHT_CSV: Path = EXTERNAL_DIR / "daylight.csv"
MACRO_CSV: Path = EXTERNAL_DIR / "macro.csv"

# География Москвы (Единый диспетчерский центр / депо)
LAT: float = 55.751244
LON: float = 37.618423
TZ: str = "Europe/Moscow"

START: str = "2025-01-01"
END: str = "2025-12-31"

# База отопительного градусо-дня: 18 °C, как в стандартах IMD
HDD_BASE_C: float = 18.0

TIMEOUT: int = 40
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mt-hackathon/1.0)"}


def _requests() -> "requests.Session":
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch_daylight(session: "requests.Session") -> Path:
    """Световой день в Москве: восход, заход и их длительность."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        "&daily=sunrise,sunset"
        f"&timezone={TZ}"
    )
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    js = r.json()
    daily = js["daily"]
    sunrise = pd.to_datetime(pd.Series(daily["sunrise"]))
    sunset = pd.to_datetime(pd.Series(daily["sunset"]))
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "sunrise": sunrise.dt.tz_localize(None),
            "sunset": sunset.dt.tz_localize(None),
        }
    )
    df["daylight_hours"] = (df["sunset"] - df["sunrise"]).dt.total_seconds() / 3600.0
    df["darkness_hours"] = 24.0 - df["daylight_hours"]
    df[["date", "daylight_hours", "darkness_hours"]].to_csv(
        DAYLIGHT_CSV, index=False, encoding="utf-8"
    )
    logger.info("Световой день сохранён: %s (строк %d)", DAYLIGHT_CSV, len(df))
    return DAYLIGHT_CSV


def _yahoo_daily(session: "requests.Session", symbol: str, start: str, end: str) -> pd.Series:
    """Дневной ряд Yahoo Finance по тикеру."""
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(
        (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=2))
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d"
    )
    r = session.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(TZ).normalize()
    idx = idx.tz_localize(None)
    vals = res["indicators"]["quote"][0]["close"]
    return pd.Series(vals, index=idx, name=symbol)


def fetch_macro(session: "requests.Session") -> Path:
    """Нефть Brent и курс USD/RUB - дневные макро-ряды."""
    brent = _yahoo_daily(session, "BZ=F", START, END)
    fx = _yahoo_daily(session, "USDRUB=X", START, END)
    df = pd.concat([brent, fx], axis=1)
    df = df.loc[(df.index >= START) & (df.index <= END)]
    # Выходные без торгов: переносим последнее значение вперёд
    df = df.ffill()
    df.index.name = "date"
    df.reset_index().to_csv(MACRO_CSV, index=False, encoding="utf-8")
    logger.info("Макро-данные сохранены: %s (строк %d)", MACRO_CSV, len(df))
    return MACRO_CSV


def add_heating_degree_days(feats: pd.DataFrame) -> pd.DataFrame:
    """
    Отопительные градусо-дни (HDD) из уже имеющейся погоды.

    Сумма (18 - t) по часам за последние 1/3/7 дней - классический
    индикатор суровости зимы, который лучше мгновенной температуры
    объясняет переход «пешком или на машине».
    """
    if "temperature_2m" not in feats.columns:
        return feats
    out = feats.sort_values(["route", "date", "hour"]).copy()
    hdd = (HDD_BASE_C - out["temperature_2m"]).clip(lower=0.0)
    out["hdd"] = hdd
    grp = out.groupby(["route", "date"])["hdd"].transform("mean")
    out["hdd_day"] = grp
    for window in (24, 72, 168):
        out[f"hdd_{window}h"] = (
            out.groupby("route")["hdd"]
            .transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        )
    out.loc[out["date"] < out["date"].min(), "hdd_day"] = 0.0
    return out


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    session = _requests()
    for name, fn in (("световой день", fetch_daylight), ("макро", fetch_macro)):
        try:
            fn(session)
        except Exception as exc:
            logger.warning("Не удалось получить %s: %s", name, exc)
    print("Готово, файлы в", EXTERNAL_DIR)


if __name__ == "__main__":
    main()
