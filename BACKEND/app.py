"""API de triaje predictivo multimodal (apoyo a la decisión clínica; no reemplaza al médico)."""
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

from config import AUC_IMAGEN_MIN, FEATURES, RANGOS

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / os.getenv("MODEL_DIR", "models")      # demo: MODEL_DIR=models_demo
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",")
MAX_IMG_BYTES = 5 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("triaje")   # NO registrar datos del paciente

state = {"clinical": None, "meta": None, "vision": None, "vmeta": None, "vision_motivo": "", "pesos": (1.0, 0.0)}


def cargar_vision():
    """El modelo de imagen es opcional y solo se activa si es mejor que el azar."""
    state.update(vision=None, vmeta=None, vision_motivo="", pesos=(1.0, 0.0))
    vpath = next((MODEL_DIR / n for n in ("vision_model.keras", "vision_model.h5")
                  if (MODEL_DIR / n).exists() and (MODEL_DIR / n).stat().st_size > 0), None)
    vmeta_path = MODEL_DIR / "vision_meta.json"
    if vpath is None or not vmeta_path.exists():
        state["vision_motivo"] = "Módulo de imagen no entrenado todavía."
        return
    vmeta = json.loads(vmeta_path.read_text())
    auc_i, auc_c = vmeta.get("auc_cv", 0.5), state["meta"].get("auc_validacion", 0.5)
    if auc_i < AUC_IMAGEN_MIN:
        state["vision_motivo"] = f"Modelo de imagen desactivado (AUC {auc_i:.2f} < {AUC_IMAGEN_MIN})."
        return
    try:
        from tensorflow import keras
        state["vision"] = keras.models.load_model(vpath)
    except Exception as e:  # noqa: BLE001
        state["vision_motivo"] = "No se pudo cargar el modelo de imagen."
        log.warning("Visión no disponible: %s", e)
        return
    # Pesos según desempeño en validación (sección 3.4.3): proporcional a (AUC - 0.5), wc + wi = 1
    ac, ai = max(auc_c - 0.5, 0.0), max(auc_i - 0.5, 0.0)
    state["pesos"] = (ac / (ac + ai), ai / (ac + ai))
    state["vmeta"] = vmeta


@asynccontextmanager
async def lifespan(app: FastAPI):
    mp, meta = MODEL_DIR / "clinical_model.pkl", MODEL_DIR / "model_meta.json"
    if not mp.exists() or not meta.exists():
        raise RuntimeError(f"Falta el modelo en {MODEL_DIR}. Ejecuta train_clinical.py primero.")
    state["clinical"], state["meta"] = joblib.load(mp), json.loads(meta.read_text())
    cargar_vision()
    log.info("Clínico: %s | visión activa: %s | pesos (wc, wi): %s | demo: %s", state["meta"]["modelo"],
             state["vision"] is not None, tuple(round(w, 2) for w in state["pesos"]), state["meta"].get("demo"))
    yield


app = FastAPI(title="EsSalud Predictive Triage API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def score_imagen(img_bytes: bytes) -> float:
    if len(img_bytes) > MAX_IMG_BYTES:
        raise HTTPException(413, "La imagen supera 5 MB.")
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Archivo de imagen inválido.")
    arr = np.expand_dims(np.asarray(img, dtype="float32"), 0)   # 0-255: el modelo ya reescala por dentro
    return float(state["vision"].predict(arr, verbose=0).ravel()[0])


def alerta_signos_vitales(spo2, pas, mental):
    """Señales de alarma que el modelo no puede 'bajar'. Valores de ejemplo: validar con un médico."""
    return spo2 < 90 or pas < 90 or mental >= 2


@app.get("/api/health")
def health():
    return {"ok": True, "modelo": state["meta"]["modelo"], "vision_activa": state["vision"] is not None,
            "vision_motivo": state["vision_motivo"], "demo": state["meta"].get("demo", False)}


def R(campo):  # límites de validación desde config
    lo, hi = RANGOS[campo]
    return {"ge": lo, "le": hi}


@app.post("/api/predict")
async def predict_risk(
    edad: float = Form(..., **R("edad")),
    sexo: int = Form(..., ge=0, le=1),
    temperatura: float = Form(..., **R("temperatura")),
    frecuencia_cardiaca: float = Form(..., **R("frecuencia_cardiaca")),
    frecuencia_respiratoria: float = Form(..., **R("frecuencia_respiratoria")),
    presion_sistolica: float = Form(..., **R("presion_sistolica")),
    presion_diastolica: float = Form(..., **R("presion_diastolica")),
    saturacion_o2: float = Form(..., **R("saturacion_o2")),
    estado_mental: int = Form(..., **R("estado_mental")),
    dolor: float = Form(..., **R("dolor")),
    foto_paciente: UploadFile = File(None),
):
    if presion_diastolica >= presion_sistolica:
        raise HTTPException(422, "La presión diastólica debe ser menor que la sistólica.")

    datos = {"edad": edad, "sexo": sexo, "temperatura": temperatura, "frecuencia_cardiaca": frecuencia_cardiaca,
             "frecuencia_respiratoria": frecuencia_respiratoria, "presion_sistolica": presion_sistolica,
             "presion_diastolica": presion_diastolica, "saturacion_o2": saturacion_o2,
             "estado_mental": estado_mental, "dolor": dolor}
    s_clin = float(state["clinical"].predict_proba(pd.DataFrame([datos])[FEATURES])[0][1])

    # R = wc·Sc + wi·Si  (solo si hay foto Y modelo de imagen activo; si no, R = Sc)
    s_img, wc, wi = None, 1.0, 0.0
    if foto_paciente is not None and state["vision"] is not None:
        s_img = score_imagen(await foto_paciente.read())
        wc, wi = state["pesos"]
    riesgo = wc * s_clin + wi * (s_img or 0.0)

    u_mod, u_grave = state["meta"]["umbral_moderado"], state["meta"]["umbral_grave"]
    nivel = "GRAVE" if riesgo >= u_grave else "MODERADO" if riesgo >= u_mod else "ESTABLE"
    alerta = alerta_signos_vitales(saturacion_o2, presion_sistolica, estado_mental)
    if alerta:
        nivel = "GRAVE"

    recomendaciones = {"GRAVE": "Priorizar atención médica inmediata.",
                       "MODERADO": "Evaluación prioritaria en triaje diferenciado.",
                       "ESTABLE": "Atención en consulta regular o control ambulatorio."}
    return {
        "score_clinico": round(s_clin, 4),
        "score_imagen": None if s_img is None else round(s_img, 4),
        "imagen_evaluada": s_img is not None,
        "pesos": {"clinico": round(wc, 3), "imagen": round(wi, 3)},
        "riesgo_ponderado": round(riesgo, 4),
        "nivel_riesgo": nivel,
        "alerta_signos_vitales": alerta,
        "recomendacion": recomendaciones[nivel],
        "umbrales": {"moderado": u_mod, "grave": u_grave},
        "modo_demo": state["meta"].get("demo", False),
        "aviso": "Herramienta de apoyo a la decisión; la evaluación final es del personal de salud.",
    }