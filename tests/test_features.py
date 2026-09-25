"""
Тесты признаков и отсутствия утечки (leakage guard).
Запуск: python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.features import get_feature_columns, make_features  # noqa: E402
from src.metrics import compute_metrics, round_predictions  # noqa: E402


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    from src.data import load_and_prepare

    return make_features(load_and_prepare(), cutoff=cfg.VALID_START, use_weather=True)


def test_feature_columns_present(feats: pd.DataFrame) -> None:
    """Все признаки из списка построены и без NaN."""
    cols = get_feature_columns(use_weather=True)
    missing = [c for c in cols if c not in feats.columns]
    assert not missing, f"нет признаков: {missing}"
    assert int(feats[cols].isna().sum().sum()) == 0


def test_exclude_removes_columns() -> None:
    """exclude действительно убирает признаки."""
    cols = get_feature_columns(
        use_weather=True, exclude=("month", "dayofyear", "weekofyear", "day")
    )
    for drop in ("month", "dayofyear", "weekofyear", "day"):
        assert drop not in cols


def test_history_aggregates_use_only_past(feats: pd.DataFrame) -> None:
    """
    Утечка: агрегаты для будущих дат не должны совпадать с фактическим
    значением boardings (иначе явная утечка ground truth).
    """
    valid = feats[feats["date"] >= cfg.VALID_START]
    # Для строк с ненулевым фактом медиана истории не должна ему равняться
    # «идеально» во всех случаях — допускаем совпадения не более 60%.
    nonzero = valid[valid["boardings"] > 0]
    exact = float(np.mean(nonzero["hist_rhd_med"].values == nonzero["boardings"].values))
    assert exact < 0.6, f"подозрительно много точных совпадений: {exact:.1%}"


def test_hist_rhd_med_is_a_past_daily_panel(feats: pd.DataFrame) -> None:
    """hist_rhd_med постоянна внутри (route, hour, dow) — это исторический срез."""
    valid = feats[feats["date"] >= cfg.VALID_START]
    nun = valid.groupby(["route", "hour", "dow"])["hist_rhd_med"].nunique()
    assert (nun <= 1).all()


def test_trend_ratio_bounds(feats: pd.DataFrame) -> None:
    """trend_ratio ограничен [0.3, 3.0] (защита от выбросов)."""
    assert feats["trend_ratio"].between(0.3, 3.0).all()


def test_metrics_wape_known_values() -> None:
    """WAPE и WAPE-score считаются корректно."""
    m = compute_metrics([100, 100], [80, 120])
    assert m["WAPE"] == 0.2
    assert m["WAPE_score"] == pytest.approx(0.8)
    assert m["MAE"] == pytest.approx(20.0)


def test_metrics_all_zero_truth() -> None:
    """Деление на ноль при пустом факте не ломает метрику."""
    m = compute_metrics([0, 0], [0, 0])
    assert m["WAPE"] == 1.0
    assert m["WAPE_score"] == 0.0


def test_round_predictions_clips_and_rounds() -> None:
    """clip(>=0) и округление до целых."""
    out = round_predictions([-5.4, 0.4, 10.5, 2.5])
    assert out.tolist() == [0, 0, 10, 2]
    assert np.issubdtype(out.dtype, np.integer)
