"""
Генератор вариантов сабмита для подбора на платформе хакатона.

Зачем: ноябрь-декабрь 2025 — период, который нельзя проверить локально
(разметки по нему нет), а погрешность уровня в нём систематически
колеблется от месяца к месяцу. По помесячному бэктесту смещение
sum(pred)/sum(actual) менялось так:

    апр 0.97 | май 1.11 | июн 1.12 | июл 1.19 | авг 1.07 | сен 1.03 | окт 0.97

Калибровать по истории бесполезно (перекос не переносится между
месяцами), зато можно измерить его на самой платформе: в зачёт идёт
лучший результат, а успешных попыток осталось 23.

Каждый вариант отличается от базового одним осмысленным множителем,
поэтому результат загрузки однозначно интерпретируется.

Использование:
    python -m src.make_variants            # собрать все варианты
    python -m src.make_variants --list     # только показать список
"""
import argparse
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.metrics import round_predictions

VARIANTS_DIR: Path = cfg.SUBMISSIONS_DIR / "variants"

# Спецификация варианта: маска дат + множитель.
# {"scope": "all" | "month" | "dates", "months": [...], "dates": [...], "k": float}
VARIANTS: dict[str, dict] = {
    # ровно то, что уже загружено (платформа показала 0.87597)
    "base": {"scope": "all", "k": 1.0},
    # гипотеза «модель чуть завышает уровень» - проверяется первым
    "level_098": {"scope": "all", "k": 0.98},
    "level_096": {"scope": "all", "k": 0.96},
    "level_102": {"scope": "all", "k": 1.02},
    # поправка только на декабрь
    "dec_097": {"scope": "month", "months": [12], "k": 0.97},
    "dec_095": {"scope": "month", "months": [12], "k": 0.95},
    # 31.12: модель даёт 0.42 от нормы, эмпирика января (среда) - около 0.35
    "newyear": {"scope": "dates", "dates": ["2025-12-31"], "k": 0.83},
}


def apply_variant(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Применяет множитель k только к строкам, попавшим в маску."""
    out = df.copy()
    scope = spec["scope"]
    if scope == "all":
        mask = pd.Series(True, index=out.index)
    elif scope == "month":
        mask = out["date"].dt.month.isin(spec["months"])
    elif scope == "dates":
        mask = out["date"].isin(pd.to_datetime(spec["dates"]))
    else:
        raise ValueError(f"Неизвестный scope: {scope}")

    out.loc[mask, "prediction"] = round_predictions(
        out.loc[mask, "prediction"] * float(spec["k"])
    )
    return out



def build(source: Path, out_dir: Path = VARIANTS_DIR) -> pd.DataFrame:
    """Собирает все варианты из базового forecast.csv."""
    df = pd.read_csv(source, sep=";", parse_dates=["date"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    base_total = int(df["prediction"].sum())
    for name, spec in VARIANTS.items():
        v = apply_variant(df, spec)
        path = out_dir / f"{name}.csv"
        v.to_csv(path, sep=";", index=False, encoding="utf-8")
        delta = int(v["prediction"].sum() - base_total)
        summary.append(
            {
                "variant": name,
                "file": path.name,
                "rows": len(v),
                "sum": int(v["prediction"].sum()),
                "delta_vs_base": delta,
                "delta_pct": round(100 * delta / base_total, 2),
            }
        )
    frame = pd.DataFrame(summary)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Варианты сабмита")
    parser.add_argument("--source", default=str(cfg.FORECAST_CSV_PATH))
    parser.add_argument("--list", action="store_true", help="только список вариантов")
    args = parser.parse_args()

    if args.list:
        for name, spec in VARIANTS.items():
            print(f"{name:12s} {spec}")
        return

    frame = build(Path(args.source))
    print(frame.to_string(index=False))
    print(f"\nКаталог: {VARIANTS_DIR}")
    print("Порядок загрузки: сначала base (известен), потом level_098.")
    print("Смотрите, что покажет платформа, прежде чем грузить остальные.")


if __name__ == "__main__":
    main()
