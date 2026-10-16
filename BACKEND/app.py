from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import io
from PIL import Image

app = FastAPI(title="EsSalud Predictive Diagnostic API")

# Habilitar CORS para permitir peticiones desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo tabular
clinical_model = joblib.load("models/clinical_model.pkl")

# Pesos de integración multimodal definidos en la tesis (wc + wi = 1)
W_CLINICAL = 0.65
W_IMAGE = 0.35

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/api/predict")
async def predict_risk(
    edad: float = Form(...),
    sexo: int = Form(...),              # 0: Femenino, 1: Masculino
    temperatura: float = Form(...),
    frecuencia_cardiaca: float = Form(...),
    presion_sistolica: float = Form(...),
    presion_diastolica: float = Form(...),
    sintoma_severidad: int = Form(...), # 0: Leve, 1: Moderado, 2: Severo
    foto_paciente: UploadFile = File(None)
):
    # 1. Score Clínico (Sc)
    input_clinico = np.array([[edad, sexo, temperatura, frecuencia_cardiaca, 
                               presion_sistolica, presion_diastolica, sintoma_severidad]])
    
    # Probabilidad de clase positiva (riesgo)
    score_clinico = float(clinical_model.predict_proba(input_clinico)[0][1])

    # 2. Score de Imagen (Si)
    score_imagen = 0.0
    if foto_paciente:
        img_bytes = await foto_paciente.read()
        _ = preprocess_image(img_bytes)
        # score_imagen = float(vision_model.predict(img_proc)[0][0])
        score_imagen = 0.5  # Valor referencial si la imagen se evalúa
        riesgo_final = (W_CLINICAL * score_clinico) + (W_IMAGE * score_imagen)
    else:
        riesgo_final = score_clinico

    # 3. Clasificación de Categoría de Riesgo
    if riesgo_final >= 0.70:
        categoria = "GRAVE"
        recomendacion = "Atención médica inmediata en Trauma Shock / Especialidad de Urgencia."
    elif riesgo_final >= 0.40:
        categoria = "MODERADO"
        recomendacion = "Evaluación prioritaria en triaje diferenciado."
    else:
        categoria = "ESTABLE"
        recomendacion = "Atención en consulta regular o control ambulatorio."

    return {
        "score_clinico": round(score_clinico, 4),
        "score_imagen": round(score_imagen, 4),
        "riesgo_ponderado": round(riesgo_final, 4),
        "nivel_riesgo": categoria,
        "recomendacion": recomendacion
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)