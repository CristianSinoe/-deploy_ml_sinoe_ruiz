from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title="Stroke Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo (puede ser dict con pipeline y threshold o el pipeline directo)
loaded = load(pathlib.Path("model/stroke-model-v1.joblib"))
if isinstance(loaded, dict):
    model = loaded["pipe"]
    THRESHOLD = loaded.get("threshold", 0.5)
else:
    model = loaded
    THRESHOLD = 0.5  # compat

# Esquema de entrada alineado al dataset original
class InputData(BaseModel):
    # numéricos / binarios
    age: float = 61
    hypertension: int = 0        # 0/1
    heart_disease: int = 0       # 0/1
    avg_glucose_level: float = 202.21
    bmi: Optional[float] = None  # puede ser None

    # categóricos (texto tal cual viene en el dataset)
    gender: str = "Female"               # "Male" | "Female" | "Other"
    ever_married: str = "Yes"            # "Yes" | "No"
    work_type: str = "Self-employed"     # "Private" | "Self-employed" | "Govt_job" | "children" | "Never_worked"
    Residence_type: str = "Rural"        # "Urban" | "Rural"
    smoking_status: str = "never smoked" # "formerly smoked" | "never smoked" | "smokes" | "Unknown"

class OutputData(BaseModel):
    score: float  # probabilidad de stroke (clase positiva)

@app.post("/score", response_model=OutputData)
def score(data: InputData):
    # Crear DataFrame con los nombres EXACTOS que el modelo espera
    payload = pd.DataFrame([{
        "age": data.age,
        "avg_glucose_level": data.avg_glucose_level,
        "bmi": data.bmi,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "gender": data.gender,
        "ever_married": data.ever_married,
        "work_type": data.work_type,
        "Residence_type": data.Residence_type,
        "smoking_status": data.smoking_status,
    }])

    # Predecir probabilidad de clase positiva
    proba = model.predict_proba(payload)[:, -1]
    return {"score": float(proba.item())}
