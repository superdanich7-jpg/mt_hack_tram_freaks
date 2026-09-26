"""
Анализ примера валидаций (Хакатон_пример_валидаций.xlsx) как источника фич.

Задача файла по ТЗ V3: проверить, можно ли получить из него дополнительные
признаки (тип карты/продукта, доля успешных валидаций по часам) и как он
соотносится с разметкой boardings.

Ключевой вопрос — пересечение по времени: разметка покрывает 2025 год,
а пример валидаций содержит единичные транзакции другой даты. Ответ
проверяется программно, а не предполагается.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VALIDATIONS_XLSX: Path = Path("spravochniki/Хакатон_пример_валидаций.xlsx")

# Колонки, ради которых файл вообще может быть полезен
COL_TRANSACTION_TIME = "Дата и время транзакции"
COL_TRIP_START = "Дата и время начала поездки"
COL_PRODUCT = "Продукт"
COL_RESULT = "Результат прохода"
COL_MEDIA = "Архитектура носителя"
COL_TARIFF = "Тариф"
COL_ROUTE = "Маршрут НГПТ"
COL_PLACE = "Место прохода"
COL_CARD = "Транспортный номер карты"


def load_validations(path: Path = VALIDATIONS_XLSX) -> pd.DataFrame:
    """Загрузка примера валидаций с разбором времени."""
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не найден.")
    df = pd.read_excel(path)
    df[COL_TRANSACTION_TIME] = pd.to_datetime(df[COL_TRANSACTION_TIME], errors="coerce")
    df[COL_TRIP_START] = pd.to_datetime(df[COL_TRIP_START], errors="coerce")
    return df


def temporal_overlap(df: pd.DataFrame) -> dict:
    """
    Пересечение по времени с разметкой boardings (2025-01-01 … 2025-10-31).

    Если пересечения нет, файл нельзя использовать как источник фич по датам.
    """
    lo, hi = df[COL_TRANSACTION_TIME].min(), df[COL_TRANSACTION_TIME].max()
    n_in = int(
        ((df[COL_TRANSACTION_TIME] >= pd.Timestamp(cfg.HISTORY_START))
         & (df[COL_TRANSACTION_TIME] <= pd.Timestamp(cfg.HISTORY_END) + pd.Timedelta(days=1))).sum()
    )
    return {
        "min_ts": str(lo),
        "max_ts": str(hi),
        "n_rows": int(len(df)),
        "n_rows_in_labels_range": n_in,
        "overlaps_labels": bool(n_in > 0),
    }


def product_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Распределение типов карт/продуктов и успешности проходов."""
    return (
        df.groupby([COL_MEDIA, COL_PRODUCT], dropna=False)
        .agg(
            n=("Проход", "size") if "Проход" in df.columns else (COL_RESULT, "size"),
            success_rate=(COL_RESULT, lambda s: float((s.astype(str).str.contains("Отказ", na=False) == False).mean())),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Распределение валидаций по часам: есть ли пики, как в boardings."""
    hours = df[COL_TRANSACTION_TIME].dt.hour
    return (
        pd.DataFrame({"hour": hours})
        .value_counts()
        .rename_axis("hour")
        .reset_index(name="n")
        .sort_values("hour")
    )


def usable_as_feature_source(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Итоговый вердикт: годится ли файл для построения фич.

    Критерии: объём, пересечение по времени с разметкой, наличие ID остановки.
    """
    ov = temporal_overlap(df)
    reasons: list[str] = []
    if len(df) < 1000:
        reasons.append(f"слишком мало строк ({len(df)} < 1000)")
    if not ov["overlaps_labels"]:
        reasons.append(
            f"нет пересечения по времени с разметкой ({ov['min_ts'][:10]} … {ov['max_ts'][:10]})"
        )
    if reasons:
        return False, "; ".join(reasons)
    return True, "файл пригоден как источник фич"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    v = load_validations()
    print("=" * 60)
    print(f"Строк: {len(v)}, колонок: {v.shape[1]}")
    print("Пересечение с разметкой:", temporal_overlap(v))
    print("-" * 60)
    print("Продукты / носители:")
    print(product_profile(v).to_string(index=False))
    print("-" * 60)
    print("Распределение по часам:")
    print(hourly_profile(v).to_string(index=False))
    print("-" * 60)
    ok, verdict = usable_as_feature_source(v)
    print(f"ВЕРДИКТ: usable={ok} — {verdict}")
    print("=" * 60)
