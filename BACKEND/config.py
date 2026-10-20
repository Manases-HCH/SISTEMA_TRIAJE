"""Configuración compartida entre entrenamiento y API (una sola fuente de verdad)."""

# Orden EXACTO de las columnas que usa el modelo clínico.
# clinical_data.csv debe tener estas columnas + la columna objetivo.
FEATURES = [
    "edad",
    "sexo",                 # 0: Femenino, 1: Masculino
    "temperatura",          # °C
    "frecuencia_cardiaca",  # lpm
    "presion_sistolica",    # mmHg
    "presion_diastolica",   # mmHg
    "saturacion_o2",        # %
    "sintoma_severidad",    # 0: Leve, 1: Moderado, 2: Severo
]
TARGET = "riesgo"  # 0: Estable, 1: Riesgo/Grave (definido por el médico/triaje final = gold standard)

# Rangos fisiológicos plausibles: sirven para limpiar el dataset y validar la API.
RANGOS = {
    "edad": (0, 120),
    "temperatura": (30, 45),
    "frecuencia_cardiaca": (20, 250),
    "presion_sistolica": (40, 300),
    "presion_diastolica": (20, 200),
    "saturacion_o2": (50, 100),
}

# Pesos multimodales (wc + wi = 1). Ajústalos con el desempeño en validación (sección 3.4.3).
W_CLINICAL = 0.65
W_IMAGE = 0.35

RANDOM_STATE = 42