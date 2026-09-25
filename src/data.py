"""
Модуль загрузки и первичной обработки разметки почасовых валидаций.
Формирует непрерывную сетку route x date x hour и сохраняет data/processed/boardings.parquet.
"""
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np

ROUTES: list[int] = [1, 7, 11, 12, 17, 25, 26, 28, 50]
HOURS: range = range(24)
START_DATE: str = "2025-01-01"
END_DATE: str = "2025-10-31"

TRAIN_LABELS_PATH: Path = Path("labels/labels_day_train.csv")
TEST_LABELS_PATH: Path = Path("labels/labels_day_test.csv")
OUTPUT_PARQUET_PATH: Path = Path("data/processed/boardings.parquet")


def load_labels(path: str | Path) -> pd.DataFrame:
    """Загрузка файла разметки с разделителем ';' и приведением типов."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл {path} не найден.")

    df = pd.read_csv(
        path,
        sep=";",
        dtype={"route": "int32", "hour": "int32", "boardings": "int32"},
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def build_full_grid(
    routes: Sequence[int] = ROUTES,
    dates: Sequence[pd.Timestamp] | None = None,
    hours: Sequence[int] = HOURS,
) -> pd.DataFrame:
    """Построение декартовой сетки маршрут x дата x час."""
    if dates is None:
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")

    grid_index = pd.MultiIndex.from_product(
        [routes, dates, hours],
        names=["route", "date", "hour"],
    )
    grid_df = pd.DataFrame(index=grid_index).reset_index()
    grid_df["route"] = grid_df["route"].astype("int32")
    grid_df["hour"] = grid_df["hour"].astype("int32")
    grid_df["date"] = pd.to_datetime(grid_df["date"])
    return grid_df


def load_and_prepare(
    train_path: str | Path = TRAIN_LABELS_PATH,
    test_path: str | Path = TEST_LABELS_PATH,
    output_path: str | Path = OUTPUT_PARQUET_PATH,
) -> pd.DataFrame:
    """Чтение labels train + test, построение полной сетки и сохранение в parquet."""
    df_train = load_labels(train_path)
    df_test = load_labels(test_path)

    df_raw = pd.concat([df_train, df_test], ignore_index=True)

    # Исключаем маршруты, которых не должно быть в валидном обучении (например 5, если появится)
    df_raw = df_raw[df_raw["route"].isin(ROUTES)].copy()

    # Дедупликация если есть повторы
    df_raw = df_raw.groupby(["route", "date", "hour"], as_index=False)["boardings"].sum()

    # Сетка
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    grid = build_full_grid(routes=ROUTES, dates=dates, hours=HOURS)

    # Объединение с сеткой, пропуски заполняются 0
    df_full = pd.merge(grid, df_raw, on=["route", "date", "hour"], how="left")
    df_full["boardings"] = df_full["boardings"].fillna(0).astype("int64")

    # Сортировка по времени и маршруту
    df_full = df_full.sort_values(["route", "date", "hour"]).reset_index(drop=True)

    # Валидация
    assert not df_full["boardings"].isna().any(), "Присутствуют NaN в boardings!"
    assert 5 not in df_full["route"].unique(), "Маршрут 5 не должен присутствовать!"
    assert len(df_full) == len(ROUTES) * len(dates) * 24, (
        f"Ожидалось {len(ROUTES) * len(dates) * 24} строк, получено {len(df_full)}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(output_path, index=False)
    return df_full


if __name__ == "__main__":
    df = load_and_prepare()
    print("=" * 60)
    print("Датасет успешно подготовлен!")
    print(f"Shape: {df.shape}")
    print(f"Диапазон дат: {df['date'].min().strftime('%Y-%m-%d')} ... {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Маршруты ({len(df['route'].unique())}): {sorted(df['route'].unique().tolist())}")
    print(f"Часы: {sorted(df['hour'].unique().tolist())}")
    print(f"Всего валидаций: {df['boardings'].sum():,}")
    print("\nСумма валидаций по месяцам:")
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["boardings"].sum()
    for m, val in monthly.items():
        print(f"  {m}: {val:,}")
    print("=" * 60)
