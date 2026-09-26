"""
Статистические базисы для почасового пассажиропотока.

Смысл: проверить, выигрывают ли деревья у классических моделей времени
(ETS с суточной/недельной сезонностью), и получить независимый прогноз
для ансамбля.

Holt-Winters (ETS) с двумя сезонностями в statsmodels не поддерживается,
поэтому используем одну — суточную (periods=24) — плюс тренд.
SARIMAX с (1,1,1)x(1,1,1,24) — как более «серьёзный» базис.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Сезонность для почасового ряда — сутки
SEASON_PERIOD: int = 24
# Порядок SARIMAX: недельная сезонность длиной 168 слишком тяжёлая, берём суточную
SARIMAX_ORDER: tuple[int, int, int] = (1, 1, 1)
SARIMAX_SEASONAL: tuple[int, int, int, int] = (1, 1, 1, SEASON_PERIOD)
# Длина обучающей выборки и число итераций оптимизатора.
# Полная история (7 000 точек × 9 маршрутов) считается минутами — берём хвост.
TRAIN_DAYS: int = 60
SARIMAX_MAXITER: int = 15

warnings.filterwarnings("ignore")


def _route_series(hist: pd.DataFrame, route: int) -> np.ndarray:
    s = (
        hist[hist["route"] == route]
        .sort_values(["date", "hour"])["boardings"]
        .astype(float)
        .to_numpy()
    )
    return s


def forecast_ets(feats: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """
    Holt-Winters (аддитивный тренд, аддитивная суточная сезонность) по каждому маршруту.

    :return: прогноз для строк feats[date >= cutoff] в порядке исходного кадра.
    """
    hist = feats[feats["date"] < cutoff]
    fut = feats[feats["date"] >= cutoff].sort_values(["route", "date", "hour"]).reset_index(drop=True)
    out = np.zeros(len(fut), dtype=float)

    for route, grp in fut.groupby("route", sort=False):
        s = _route_series(hist, route)
        if len(s) < 2 * SEASON_PERIOD:
            out[grp.index] = 0.0
            continue
        try:
            model = ExponentialSmoothing(
                s, trend="add", seasonal="add", seasonal_periods=SEASON_PERIOD
            )
            fit = model.fit(optimized=True, use_brute=False)
            out[grp.index] = fit.forecast(len(grp))
        except Exception as exc:
            logger.warning("ETS не сошёлся для route=%s: %s", route, exc)
            out[grp.index] = s[-SEASON_PERIOD:].mean()
    return out


def forecast_sarimax(feats: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """
    SARIMAX (1,1,1)x(1,1,1,24) на последних TRAIN_DAYS каждого маршрута.

    Полная история слишком длинная, поэтому используем свежий хвост.
    """
    hist = feats[feats["date"] < cutoff]
    fut = feats[feats["date"] >= cutoff].sort_values(["route", "date", "hour"]).reset_index(drop=True)
    out = np.zeros(len(fut), dtype=float)
    start = cutoff - pd.Timedelta(days=TRAIN_DAYS)

    for route, grp in fut.groupby("route", sort=False):
        s = _route_series(hist[hist["date"] >= start], route)
        if len(s) < 3 * SEASON_PERIOD:
            out[grp.index] = 0.0
            continue
        try:
            model = SARIMAX(
                s, order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL, enforce_stationarity=False
            )
            fit = model.fit(disp=False, maxiter=SARIMAX_MAXITER)
            out[grp.index] = fit.forecast(len(grp))
        except Exception as exc:
            logger.warning("SARIMAX не сошёлся для route=%s: %s", route, exc)
            out[grp.index] = s[-SEASON_PERIOD:].mean()
    return out


def seasonal_naive(feats: pd.DataFrame, cutoff: pd.Timestamp) -> np.ndarray:
    """
    Наивный сезонный прогноз: значение того же часа недели назад (168 часов).

    Требует, чтобы в фрейме были строки с boardings; для будущих дат берём
    последнюю доступную неделю, поэтому работает только на прокси-сплитах.
    """
    hist = feats[feats["date"] < cutoff].sort_values(["route", "date", "hour"])
    fut = feats[feats["date"] >= cutoff].sort_values(["route", "date", "hour"]).reset_index(drop=True)
    last = hist[hist["date"] >= cutoff - pd.Timedelta(days=7)]
    pivot = last.pivot_table(index=["route", "hour"], columns="date", values="boardings", aggfunc="mean")
    med = pivot.median(axis=1)
    lookup = med.reindex(
        pd.MultiIndex.from_frame(fut[["route", "hour"]].drop_duplicates())
    )
    key = pd.MultiIndex.from_frame(fut[["route", "hour"]])
    return lookup.reindex(key).to_numpy(dtype=float)


if __name__ == "__main__":
    import pandas as pd

    from src.data import load_and_prepare
    from src.features import make_features
    from src.metrics import compute_metrics, round_predictions

    df = load_and_prepare()
    cut = pd.Timestamp(cfg.VALID_START)
    feats = make_features(df, cutoff=cut, use_weather=True)
    for fname, fn in [
        ("ETS", forecast_ets),
        ("SARIMAX", forecast_sarimax),
        ("seasonal_naive", seasonal_naive),
    ]:
        preds = round_predictions(fn(feats, cut))
        fut = feats[feats["date"] >= cut]
        m = compute_metrics(fut["boardings"], preds)
        print(f"{fname:14s} WAPE-score={m['WAPE_score']:.5f} MAE={m['MAE']:.2f}")

