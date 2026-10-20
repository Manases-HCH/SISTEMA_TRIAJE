"""API de triaje predictivo (apoyo a la decisión clínica; no reemplaza el criterio médico)."""
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from config import FEATURES, RANGOS, W_CLINICAL, W_IMAGE

BASE = Path(__file__).resolve().parent
# Para probar con el modelo demo:  MODEL_DIR=models_demo uvicorn app:app
MODEL_DIR = BASE / os.getenv("MODEL_DIR", "models")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",")
MAX_IMG_BYTES = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("triaje")  # NO registrar datos del paciente en los logs

state = {"clinical": None, "meta": None, "vision": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path, meta_path = MODEL_DIR / "clinical_model.pkl", MODEL_DIR / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Falta el modelo en {MODEL_DIR}. Ejecuta train_clinical.py primero.")
    state["clinical"] = joblib.load(model_path)
    state["meta"] = json.loads(meta_path.read_text())
    # El modelo de visión es opcional: solo se usa si existe y no está vacío.
    vpath = MODEL_DIR / "vision_model.h5"
    if vpath.exists() and vpath.stat().st_size > 0:
        try:
            from tensorflow.keras.models import load_model
            state["vision"] = load_model(vpath)
        except Exception as e:  # noqa: BLE001
            log.warning("Modelo de visión no disponible: %s", e)
    log.info("Modelo: %s | visión: %s | demo: %s", state["meta"]["modelo"],
             state["vision"] is not None, state["meta"].get("demo"))
    yield


app = FastAPI(title="EsSalud Predictive Triage API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def score_imagen(img_bytes: bytes):
    if len(img_bytes) > MAX_IMG_BYTES:
        raise HTTPException(413, "La imagen supera 5 MB.")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Archivo de imagen inválido.")
    arr = np.expand_dims(np.asarray(img, dtype="float32") / 255.0, axis=0)
    return float(state["vision"].predict(arr, verbose=0)[0][0])


def reglas_de_seguridad(spo2, pas):
    """Señales de alarma que el modelo NO puede 'bajar'. Valores de ejemplo: validar con un médico."""
    return spo2 < 90 or pas < 90


@app.get("/api/health")
def health():
    return {"ok": True, "modelo": state["meta"]["modelo"], "vision_activa": state["vision"] is not None,
            "demo": state["meta"].get("demo", False)}


@app.post("/api/predict")
async def predict_risk(
    edad: float = Form(..., ge=RANGOS["edad"][0], le=RANGOS["edad"][1]),
    sexo: int = Form(..., ge=0, le=1),
    temperatura: float = Form(..., ge=RANGOS["temperatura"][0], le=RANGOS["temperatura"][1]),
    frecuencia_cardiaca: float = Form(..., ge=RANGOS["frecuencia_cardiaca"][0], le=RANGOS["frecuencia_cardiaca"][1]),
    presion_sistolica: float = Form(..., ge=RANGOS["presion_sistolica"][0], le=RANGOS["presion_sistolica"][1]),
    presion_diastolica: float = Form(..., ge=RANGOS["presion_diastolica"][0], le=RANGOS["presion_diastolica"][1]),
    saturacion_o2: float = Form(..., ge=RANGOS["saturacion_o2"][0], le=RANGOS["saturacion_o2"][1]),
    sintoma_severidad: int = Form(..., ge=0, le=2),
    foto_paciente: UploadFile = File(None),
):
    if presion_diastolica >= presion_sistolica:
        raise HTTPException(422, "La presión diastólica debe ser menor que la sistólica.")

    # DataFrame con nombres de columna: evita errores de orden de variables
    fila = pd.DataFrame([{
        "edad": edad, "sexo": sexo, "temperatura": temperatura,
        "frecuencia_cardiaca": frecuencia_cardiaca, "presion_sistolica": presion_sistolica,
        "presion_diastolica": presion_diastolica, "saturacion_o2": saturacion_o2,
        "sintoma_severidad": sintoma_severidad,
    }])[FEATURES]
    s_clin = float(state["clinical"].predict_proba(fila)[0][1])

    # Imagen: solo si hay foto Y modelo de visión real. Si no, se usa solo el score clínico.
    s_img, usa_imagen = None, False
    if foto_paciente is not None and state["vision"] is not None:
        s_img = score_imagen(await foto_paciente.read())
        usa_imagen = True
    riesgo = W_CLINICAL * s_clin + W_IMAGE * s_img if usa_imagen else s_clin

    u_mod, u_grave = state["meta"]["umbral_moderado"], state["meta"]["umbral_grave"]
    if riesgo >= u_grave:
        nivel = "GRAVE"
    elif riesgo >= u_mod:
        nivel = "MODERADO"
    else:
        nivel = "ESTABLE"

    alerta = reglas_de_seguridad(saturacion_o2, presion_sistolica)
    if alerta and nivel != "GRAVE":
        nivel = "GRAVE"

    recomendaciones = {
        "GRAVE": "Priorizar atención médica inmediata.",
        "MODERADO": "Evaluación prioritaria en triaje diferenciado.",
        "ESTABLE": "Atención en consulta regular o control ambulatorio.",
    }
    return {
        "score_clinico": round(s_clin, 4),
        "score_imagen": None if s_img is None else round(s_img, 4),
        "imagen_evaluada": usa_imagen,
        "riesgo_ponderado": round(riesgo, 4),
        "nivel_riesgo": nivel,
        "alerta_signos_vitales": alerta,
        "recomendacion": recomendaciones[nivel],
        "umbrales": {"moderado": u_mod, "grave": u_grave},
        "modo_demo": state["meta"].get("demo", False),
        "aviso": "Herramienta de apoyo a la decisión; la evaluación final es del personal de salud.",
    }