"""
Модуль выгрузки и кэширования исторических и прогнозных погодных данных Open-Meteo.
Локация: Москва (55.7558 N, 37.6173 E).
"""
import json
from pathlib import Path
import ssl
import urllib.request
import pandas as pd

WEATHER_CSV_PATH: Path = Path("data/external/weather.csv")
LATITUDE: float = 55.7558
LONGITUDE: float = 37.6173
TIMEZONE: str = "Europe/Moscow"
HOURLY_VARS: list[str] = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "cloud_cover",
    "relative_humidity_2m",
]


def fetch_weather(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    output_path: str | Path = WEATHER_CSV_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """Выгрузка погоды с Open-Meteo Archive API с кэшированием в CSV."""
    output_path = Path(output_path)
    if output_path.exists() and not force_download:
        df = pd.read_csv(output_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    hourly_str = ",".join(HOURLY_VARS)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LATITUDE}&longitude={LONGITUDE}&"
        f"start_date={start_date}&end_date={end_date}&"
        f"hourly={hourly_str}&timezone={TIMEZONE}"
    )

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Ошибка загрузки погоды с Open-Meteo: {e}")
        raise

    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    # Время приходит в виде 'YYYY-MM-DDTHH:MM'
    dt_series = pd.to_datetime(df["time"])
    df["date"] = dt_series.dt.floor("D")
    df["hour"] = dt_series.dt.hour.astype("int32")
    df = df.drop(columns=["time"])

    # Упорядочим колонки
    cols = ["date", "hour"] + [col for col in df.columns if col not in ["date", "hour"]]
    df = df[cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Погодные данные сохранены в {output_path} ({len(df)} строк).")
    return df


if __name__ == "__main__":
    df = fetch_weather(force_download=True)
    print("Пример данных о погоде:")
    print(df.head())
    print("Статистика температуры:")
    print(df["temperature_2m"].describe())
