from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

# Allow React app (running on localhost:5173 or 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + processor
model = joblib.load("model.pkl")
processor = joblib.load("processor.pkl")


class InputData(BaseModel):
    day_of_week: str
    sleep_hours: float
    study_hours: float
    break_hours: float
    start_time: str
    device_usage_minutes: float
    study_sessions: int
    noise_level: str
    social_interactions: int
    motivation: float
    prev_score: float
    notes_taken: int
    used_pomodoro: int
    material: str


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(data: InputData):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    # Transform using your preprocessor
    X = processor.transform(df)
    # Predict
    pred = model.predict(X)[0]

    try:
        pred_val = float(pred)
    except Exception:
        pred_val = pred.item()

    return {"prediction": pred_val}
