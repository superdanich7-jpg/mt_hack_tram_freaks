import secrets
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

@app.get("/routes")
def get_routes():
    return MOCK_ROUTES

@app.get("/forecast")
def get_forecast(route: str = "1", date: str = "2025-11-01"):
    return {"route": route, "date": date, "forecast": MOCK_FORECAST}

@app.get("/forecast/week")
def get_forecast_week(route: str = "1", start: str = "2025-11-01"):
    return {"route": route, "start": start, "forecast": MOCK_FORECAST}

@app.get("/forecast/month")
def get_forecast_month(route: str = "1", month: str = "2025-11"):
    return {"route": route, "month": month, "forecast": MOCK_FORECAST}
