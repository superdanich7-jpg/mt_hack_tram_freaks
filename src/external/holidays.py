"""
Модуль разметки праздничных и предпраздничных дней производственного календаря РФ за 2025 год.
"""
from pathlib import Path
import pandas as pd

HOLIDAYS_CSV_PATH: Path = Path("data/external/holidays.csv")

# Официальные нерабочие праздничные дни РФ в 2025 году
# Новогодние каникулы: 1-8 января
# День защитника Отечества: 22-23 февраля
# Международный женский день: 8-9 марта
# Майские праздники: 1-4 мая, 8-11 мая
# День России: 12-15 июня
# День народного единства: 2-4 ноября (включая перенос)
# 31 декабря - выходной день
HOLIDAYS_2025: set[str] = {
    # Январь
    "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
    "2025-01-05", "2025-01-06", "2025-01-07", "2025-01-08",
    # Февраль
    "2025-02-22", "2025-02-23",
    # Март
    "2025-03-08", "2025-03-09",
    # Май
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04",
    "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11",
    # Июнь
    "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15",
    # Ноябрь
    "2025-11-02", "2025-11-03", "2025-11-04",
    # Декабрь
    "2025-12-31",
}

# Предпраздничные дни (сокращенные рабочие дни)
PREHOLIDAYS_2025: set[str] = {
    "2025-03-07",
    "2025-04-30",
    "2025-06-11",
    "2025-11-01",
    "2025-12-30",
}


def build_holidays_calendar(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    output_path: str | Path = HOLIDAYS_CSV_PATH,
) -> pd.DataFrame:
    """Генерация таблицы праздников РФ со счетчиками дней до/после."""
    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    df["is_official_holiday"] = df["date_str"].isin(HOLIDAYS_2025).astype("int32")
    df["is_preholiday"] = df["date_str"].isin(PREHOLIDAYS_2025).astype("int32")

    # Дней до ближайшего праздника и дней после
    holiday_dates = [pd.to_datetime(d) for d in sorted(HOLIDAYS_2025)]

    def calc_dist_to_holiday(d: pd.Timestamp) -> int:
        fut = [h for h in holiday_dates if h >= d]
        if not fut:
            return 99
        return min((h - d).days for h in fut)

    def calc_dist_after_holiday(d: pd.Timestamp) -> int:
        past = [h for h in holiday_dates if h <= d]
        if not past:
            return 99
        return min((d - h).days for h in past)

    df["days_to_holiday"] = df["date"].apply(calc_dist_to_holiday).astype("int32")
    df["days_after_holiday"] = df["date"].apply(calc_dist_after_holiday).astype("int32")
    df = df.drop(columns=["date_str"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Календарь праздников сохранен в {output_path} ({len(df)} дней).")
    return df


if __name__ == "__main__":
    df = build_holidays_calendar()
    print(df[(df["date"] >= "2025-11-01") & (df["date"] <= "2025-11-06")])
    print(df[df["date"] == "2025-12-31"])
