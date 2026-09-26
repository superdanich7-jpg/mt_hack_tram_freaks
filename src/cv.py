"""
Скользящее временное разделение (rolling window) для честной валидации.

Схема из ai/ML_PIPELINE.md и ТЗ V3:
- train — всё, что строго раньше valid_start;
- valid — горизонт не менее 2 недель (чтобы попасть полный цикл будни/выходные);
- test — октябрь 2025 (самый поздний доступный месяц).

Горизонт прогноза в задаче — 2 месяца (ноябрь–декабрь) без доступа к «свежим»
данным, поэтому валидация намеренно длинная: 2 недели на валидацию + месяц на тест.
"""
from dataclasses import dataclass
import pandas as pd

# Длительность валидационного окна: 14 дней = 2 полных недели (с ТЗ V3)
VALID_HORIZON_DAYS: int = 14

# Границы скользящих фолдов и тестового периода (по ТЗ V3)
FOLD_STARTS: tuple[str, ...] = ("2025-08-01", "2025-09-01")
TEST_START: str = "2025-10-01"
TEST_END: str = "2025-10-31"


@dataclass(frozen=True)
class Fold:
    """Один временной срез: обучение, валидация и (опционально) тест."""

    name: str
    valid_start: str
    valid_end: str
    is_test: bool = False

    def masks(self, feats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Маски train/valid для кадра с признаками по колонке date."""
        train_mask = feats["date"] < self.valid_start
        valid_mask = (feats["date"] >= self.valid_start) & (feats["date"] <= self.valid_end)
        return train_mask, valid_mask


def build_folds() -> list[Fold]:
    """
    Список фолдов: два скользящих валидационных окна + тестовый месяц.

    Валидация и тест всегда позже обучения — это гарантирует отсутствие
    утечки будущего в обучающих агрегатах.
    """
    folds: list[Fold] = []
    for start in FOLD_STARTS:
        end = (
            pd.Timestamp(start) + pd.Timedelta(days=VALID_HORIZON_DAYS - 1)
        ).strftime("%Y-%m-%d")
        folds.append(Fold(name=f"valid_{start[:7]}", valid_start=start, valid_end=end))
    folds.append(
        Fold(name="test_2025-10", valid_start=TEST_START, valid_end=TEST_END, is_test=True)
    )
    return folds


def split_dates(fold: Fold) -> str:
    """Человекочитаемое описание среза для отчётов."""
    kind = "test" if fold.is_test else "valid"
    return f"{kind}: train < {fold.valid_start} | {fold.valid_start} … {fold.valid_end}"


if __name__ == "__main__":
    for f in build_folds():
        print(f"{f.name:16s} {split_dates(f)}")
