"""API: sirve el modelo champion de MLflow con FastAPI."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from config.config import FEATURES, MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME

logger = logging.getLogger("uvicorn.error")

# Estado de la API: el modelo cargado y sus datos
state = {"model": None, "info": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar, carga el modelo champion desde el Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        state["info"] = {
            "model_name": MODEL_NAME,
            "version": str(version.version),
            "alias": MODEL_ALIAS,
            "run_id": version.run_id,
            "description": version.description,
            "rmse": version.tags.get("rmse"),
            "r2": version.tags.get("r2"),
        }
        logger.info("Modelo cargado: %s versión %s (@%s)", MODEL_NAME, version.version, MODEL_ALIAS)
    except Exception as e:
        logger.error("No se pudo cargar el modelo: %s", e)
    yield
    state["model"] = None


app = FastAPI(
    title="API de predicción de diabetes",
    description="Sirve el modelo champion registrado en MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    """Datos de un paciente (valores reales, sin escalar)."""

    age: float = Field(gt=0, description="Edad en años")
    sex: Literal[1, 2] = Field(description="Sexo (1 o 2)")
    bmi: float = Field(gt=0, description="Índice de masa corporal")
    bp: float = Field(gt=0, description="Tensión arterial media")
    s1: float = Field(gt=0, description="Colesterol total")
    s2: float = Field(gt=0, description="LDL")
    s3: float = Field(gt=0, description="HDL")
    s4: float = Field(gt=0, description="Colesterol total / HDL")
    s5: float = Field(gt=0, description="Triglicéridos (log)")
    s6: float = Field(gt=0, description="Glucosa")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"age": 59, "sex": 2, "bmi": 32.1, "bp": 101, "s1": 157,
                 "s2": 93.2, "s3": 38, "s4": 4, "s5": 4.8598, "s6": 87}
            ]
        }
    )


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str


@app.get("/health")
def health():
    """Comprueba que el servicio está vivo."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info():
    """Devuelve qué modelo se está sirviendo."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return state["info"]


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """Devuelve la predicción del modelo champion para un paciente."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        data = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
        prediction = state["model"].predict(data)
    except Exception as e:
        logger.error("Error al predecir: %s", e)
        raise HTTPException(status_code=500, detail="Error al generar la predicción")
    return {"prediction": float(prediction[0]), "model_version": state["info"]["version"]}