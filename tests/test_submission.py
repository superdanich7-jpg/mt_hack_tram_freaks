"""
Тесты формата сабмита (ai/CONVENTIONS.md: test_submission.py).
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
from src.validate_submission import EXPECTED_ROWS, validate  # noqa: E402

SUBMISSION = Path(cfg.SUBMISSION_CSV_PATH)


@pytest.fixture(scope="module")
def submission() -> pd.DataFrame:
    if not SUBMISSION.exists():
        pytest.skip("submissions/submission.csv не сгенерирован — сначала `python -m src.predict`")
    return pd.read_csv(SUBMISSION, sep=";")


def test_submission_validates() -> None:
    """Полный набор проверок ai/SUBMISSION.md проходит."""
    if not SUBMISSION.exists():
        pytest.skip("submissions/submission.csv не сгенерирован")
    assert validate(SUBMISSION) is True


def test_columns(submission: pd.DataFrame) -> None:
    assert list(submission.columns) == ["route", "date", "hour", "prediction"]


def test_rows_count(submission: pd.DataFrame) -> None:
    assert len(submission) == EXPECTED_ROWS == 14640


def test_route5_zeros(submission: pd.DataFrame) -> None:
    assert (submission.loc[submission["route"] == 5, "prediction"] == 0).all()


def test_prediction_dtype_and_range(submission: pd.DataFrame) -> None:
    assert pd.api.types.is_integer_dtype(submission["prediction"])
    assert not submission["prediction"].isna().any()
    assert (submission["prediction"] >= 0).all()


def test_full_grid(submission: pd.DataFrame) -> None:
    assert submission["route"].nunique() == 10
    assert submission["date"].nunique() == 61
    assert set(submission["hour"].unique()) == set(range(24))
    assert not submission.duplicated(["route", "date", "hour"]).any()


def _date_field(line: str) -> str:
    """Второе поле строки CSV сабмита (дата)."""
    return line.split(";")[1]


def test_date_format_matches_template() -> None:
    """Формат даты берётся из test_submission.csv, а не из README."""
    if not SUBMISSION.exists() or not Path(cfg.BASELINE_SUBMISSION_PATH).exists():
        pytest.skip("нет шаблона или сабмита")
    head = SUBMISSION.read_text(encoding="utf-8").splitlines()[1]
    tmpl_head = Path(cfg.BASELINE_SUBMISSION_PATH).read_text(encoding="utf-8").splitlines()[1]
    got, expected = _date_field(head), _date_field(tmpl_head)
    assert ("." in got) == ("." in expected)
    assert len(got) == len(expected)
