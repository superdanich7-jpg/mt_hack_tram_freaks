"""
Тесты загрузки и подготовки данных (ai/CONVENTIONS.md: test_data.py).
Запуск: python -m pytest tests -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data import build_full_grid, load_and_prepare  # noqa: E402


@pytest.fixture(scope="module")
def boardings() -> pd.DataFrame:
    parquet = Path(cfg.DATA_PARQUET_PATH)
    if parquet.exists():
        return pd.read_parquet(parquet)
    return load_and_prepare()


def test_grid_cartesian() -> None:
    """Полная сетка строит декартово произведение ключей."""
    grid = build_full_grid(routes=[1, 7], dates=pd.date_range("2025-11-01", periods=2), hours=[0, 1, 2])
    assert len(grid) == 2 * 2 * 3
    assert not grid.duplicated(["route", "date", "hour"]).any()


def test_no_route_5(boardings: pd.DataFrame) -> None:
    """route=5 не должен попадать в обучающие данные."""
    assert 5 not in set(boardings["route"].unique())


def test_routes_match_labels(boardings: pd.DataFrame) -> None:
    """Набор маршрутов совпадает с labels/*."""
    assert set(boardings["route"].unique()) == set(cfg.ROUTES)


def test_no_nan_boardings(boardings: pd.DataFrame) -> None:
    """В boardings нет пропусков."""
    assert not boardings["boardings"].isna().any()


def test_full_grid_rows(boardings: pd.DataFrame) -> None:
    """Число строк = маршруты × дни × 24 часа."""
    n_dates = boardings["date"].nunique()
    assert len(boardings) == len(cfg.ROUTES) * n_dates * 24


def test_boardings_non_negative(boardings: pd.DataFrame) -> None:
    """boardings — неотрицательные целые."""
    assert (boardings["boardings"] >= 0).all()
    assert pd.api.types.is_integer_dtype(boardings["boardings"])


def test_date_range(boardings: pd.DataFrame) -> None:
    """История покрывает 2025-01-01 … 2025-10-31."""
    assert str(boardings["date"].min().date()) == cfg.HISTORY_START
    assert str(boardings["date"].max().date()) == cfg.HISTORY_END
