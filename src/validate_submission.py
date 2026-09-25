"""
Валидация файла сабмита по чеклисту ai/SUBMISSION.md.

Использование:
    python -m src.validate_submission
    python -m src.validate_submission --path submissions/submission.csv

Печатает список проверок и завершается кодом 1, если есть ошибки.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config as cfg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPECTED_ROUTES = set(cfg.ALL_ROUTES)
EXPECTED_HOURS = set(cfg.HOURS)
EXPECTED_ROWS = len(cfg.ALL_ROUTES) * 61 * 24  # 14 640


class Checker:
    """Простейший накопитель результатов проверок."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.results.append((bool(ok), name, detail))
        return bool(ok)

    def report(self) -> bool:
        print("=" * 72)
        for ok, name, detail in self.results:
            mark = "OK  " if ok else "FAIL"
            print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        n_ok = sum(1 for ok, _, _ in self.results if ok)
        print("-" * 72)
        print(f"Пройдено: {n_ok}/{len(self.results)}")
        print("OK" if n_ok == len(self.results) else "ЕСТЬ ОШИБКИ")
        print("=" * 72)
        return n_ok == len(self.results)


def validate(path: Path, template_path: Path = cfg.BASELINE_SUBMISSION_PATH) -> bool:
    c = Checker()

    c.check(path.exists(), "Файл существует", str(path))
    if not path.exists():
        return c.report()

    raw = path.read_text(encoding="utf-8")
    lines = raw.strip().splitlines()
    c.check(lines[0].strip() == "route;date;hour;prediction",
            "Заголовок = route;date;hour;prediction", lines[0].strip())
    c.check(";" in lines[1], "Разделитель ';'")

    try:
        df = pd.read_csv(path, sep=";")
        loaded = True
    except Exception as exc:  # pragma: no cover - защита от битого файла
        df = pd.DataFrame()
        loaded = False
        c.check(False, "Читается pandas без ошибок", str(exc))
    if loaded:
        c.check(True, "Читается pandas без ошибок")

    if not loaded or df.empty:
        return c.report()

    c.check(list(df.columns) == ["route", "date", "hour", "prediction"],
            "Колонки ровно: route;date;hour;prediction", str(list(df.columns)))
    c.check(len(df) == EXPECTED_ROWS, f"Строк ровно {EXPECTED_ROWS}", str(len(df)))

    # --- route ---
    c.check(pd.api.types.is_integer_dtype(df["route"]), "route — int",
            str(df["route"].dtype))
    c.check(set(df["route"].unique()) == EXPECTED_ROUTES,
            "Все 10 маршрутов [1,5,7,11,12,17,25,26,28,50]",
            str(sorted(df["route"].unique())))

    # --- date ---
    dates_str = df["date"].astype(str)
    fmt_ok = dates_str.str.fullmatch(r"\d{4}-\d{2}-\d{2}|(\d{2}\.\d{2}\.\d{4})").all()
    c.check(fmt_ok, "Формат даты единообразный", str(dates_str.iloc[0]))
    if template_path.exists():
        tmpl = pd.read_csv(template_path, sep=";", dtype={"date": str})
        c.check(dates_str.iloc[0] == str(tmpl["date"].iloc[0]),
                "Формат даты совпадает с test_submission.csv",
                f"{dates_str.iloc[0]} vs {tmpl['date'].iloc[0]}")

    dates = pd.to_datetime(dates_str, format="mixed", dayfirst="." in dates_str.iloc[0])
    c.check(dates.notna().all(), "Все даты валидны")
    c.check(str(dates.min().date()) == cfg.FORECAST_START,
            f"Первая дата = {cfg.FORECAST_START}", str(dates.min().date()))
    c.check(str(dates.max().date()) == cfg.FORECAST_END,
            f"Последняя дата = {cfg.FORECAST_END}", str(dates.max().date()))
    c.check(dates.nunique() == 61, "Ровно 61 уникальная дата (ноябрь+декабрь)", str(dates.nunique()))

    # --- hour ---
    c.check(pd.api.types.is_integer_dtype(df["hour"]), "hour — int", str(df["hour"].dtype))
    c.check(set(df["hour"].unique()) == EXPECTED_HOURS, "Часы 0…23",
            str(sorted(df["hour"].unique())))

    # --- полная сетка ---
    pairs = df.groupby(["route", "date"])["hour"].nunique()
    c.check((pairs == 24).all(), "Все 24 часа на каждую пару (route, date)")
    c.check(not df.duplicated(["route", "date", "hour"]).any(),
            "Нет дубликатов (route, date, hour)")
    grid = len(df["route"].unique()) * dates.nunique() * 24
    c.check(len(df) == grid, "Полная сетка без пропусков", f"{len(df)} = {grid}")

    # --- prediction ---
    c.check(pd.api.types.is_integer_dtype(df["prediction"]), "prediction — int",
            str(df["prediction"].dtype))
    c.check(not df["prediction"].isna().any(), "Нет NaN в prediction")
    c.check((df["prediction"] >= 0).all(), "prediction >= 0")
    r5 = df.loc[df["route"] == 5, "prediction"]
    c.check((r5 == 0).all(), "route=5 полностью нулевой", f"max={r5.max()}")

    # --- порядок строк как в шаблоне ---
    if template_path.exists():
        tmpl = pd.read_csv(template_path, sep=";", dtype={"date": str})
        same_order = (
            len(tmpl) == len(df)
            and (tmpl["route"].values == df["route"].values).all()
            and (tmpl["date"].astype(str).values == dates_str.values).all()
            and (tmpl["hour"].values == df["hour"].values).all()
        )
        c.check(same_order, "Порядок строк совпадает с test_submission.csv")

    return c.report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Валидация файла сабмита")
    parser.add_argument("--path", type=str, default=str(cfg.SUBMISSION_CSV_PATH))
    parser.add_argument("--template", type=str, default=str(cfg.BASELINE_SUBMISSION_PATH))
    args = parser.parse_args()

    ok = validate(Path(args.path), Path(args.template))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
