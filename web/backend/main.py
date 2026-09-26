import os
import secrets
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "hackathon2025")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(dependencies=[Depends(authenticate)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_ROUTES = [1, 2, 3]
MOCK_FORECAST = [{"datetime": "2025-11-01T06:00:00", "passengers": 42}]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_filepath() -> Optional[str]:
    """
    Определяет путь к файлу прогнозов:
    сначала проверяет ../../submissions/forecast.csv, если нет — ../../test_submission.csv.
    """
    candidates = [
        os.path.normpath(os.path.join(BASE_DIR, "../../submissions/forecast.csv")),
        os.path.normpath(os.path.join(BASE_DIR, "../../test_submission.csv")),
        os.path.normpath("submissions/forecast.csv"),
        os.path.normpath("test_submission.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

_CACHED_DF: Optional[pd.DataFrame] = None
_CACHED_MTIME: Optional[float] = None
_CACHED_PATH: Optional[str] = None

def load_data() -> pd.DataFrame:
    """
    Загружает DataFrame с кэшированием и обработкой ошибок:
    если файл не найден или пуст, возвращает пустой DataFrame.
    """
    global _CACHED_DF, _CACHED_MTIME, _CACHED_PATH
    filepath = get_data_filepath()
    if not filepath or not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        mtime = os.path.getmtime(filepath)
        if _CACHED_DF is not None and _CACHED_PATH == filepath and _CACHED_MTIME == mtime:
            return _CACHED_DF

        df = pd.read_csv(filepath, sep=';')
        if df.empty or "route" not in df.columns:
            return pd.DataFrame()

        df['route'] = df['route'].astype(str)
        if 'prediction' in df.columns:
            df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce').fillna(0).astype(int)

        _CACHED_DF = df
        _CACHED_MTIME = mtime
        _CACHED_PATH = filepath
        return _CACHED_DF
    except Exception:
        return pd.DataFrame()

def _to_forecast_records(subset: pd.DataFrame) -> list[dict]:
    return [
        {
            "datetime": f"{row['date']}T{int(row['hour']):02d}:00:00",
            "passengers": int(row['prediction'])
        }
        for _, row in subset.iterrows()
    ]

@app.get("/routes")
def get_routes():
    df = load_data()
    if df.empty or "route" not in df.columns:
        return MOCK_ROUTES
    unique_routes = df["route"].unique()
    try:
        return sorted([int(r) for r in unique_routes if r.isdigit()])
    except Exception:
        return sorted(list(unique_routes))

@app.get("/forecast")
def get_forecast(route: str = "1", date: str = "2025-11-01"):
    df = load_data()
    if df.empty:
        return {"route": route, "date": date, "forecast": MOCK_FORECAST}

    try:
        subset = df[(df['route'] == str(route)) & (df['date'] == str(date))]
    except Exception:
        subset = pd.DataFrame()

    if subset.empty:
        return {"route": route, "date": date, "forecast": MOCK_FORECAST}
    return {"route": route, "date": date, "forecast": _to_forecast_records(subset.sort_values(by="hour"))}

@app.get("/forecast/week")
def get_forecast_week(route: str = "1", start: str = "2025-11-01"):
    df = load_data()
    if df.empty:
        return {"route": route, "start": start, "forecast": MOCK_FORECAST}

    try:
        start_dt = pd.to_datetime(start)
        end_dt = start_dt + pd.Timedelta(days=7)
        date_series = pd.to_datetime(df['date'], errors='coerce')
        subset = df[(df['route'] == str(route)) & (date_series >= start_dt) & (date_series < end_dt)]
    except Exception:
        subset = pd.DataFrame()

    if subset.empty:
        return {"route": route, "start": start, "forecast": MOCK_FORECAST}
    return {"route": route, "start": start, "forecast": _to_forecast_records(subset.sort_values(by=["date", "hour"]))}

@app.get("/forecast/month")
def get_forecast_month(route: str = "1", month: str = "2025-11"):
    df = load_data()
    if df.empty:
        return {"route": route, "month": month, "forecast": MOCK_FORECAST}

    try:
        subset = df[(df['route'] == str(route)) & (df['date'].astype(str).str.startswith(str(month)))]
    except Exception:
        subset = pd.DataFrame()

    if subset.empty:
        return {"route": route, "month": month, "forecast": MOCK_FORECAST}
    return {"route": route, "month": month, "forecast": _to_forecast_records(subset.sort_values(by=["date", "hour"]))}

