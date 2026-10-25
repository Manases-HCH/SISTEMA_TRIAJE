"""Configuración compartida entre entrenamiento y API (una sola fuente de verdad)."""

# Orden EXACTO de las columnas del modelo clínico (clinical_data.csv = estas + TARGET).
FEATURES = [
    "edad",
    "sexo",                    # 0: Femenino, 1: Masculino
    "temperatura",             # °C
    "frecuencia_cardiaca",     # lpm
    "frecuencia_respiratoria", # rpm
    "presion_sistolica",       # mmHg
    "presion_diastolica",      # mmHg
    "saturacion_o2",           # %
    "estado_mental",           # 0: Alerta, 1: Responde a la voz, 2: Responde al dolor, 3: Inconsciente
    "dolor",                   # escala numérica 0-10
]
TARGET = "riesgo"  # 0: Estable (KTAS 4-5), 1: Riesgo (KTAS 1-3). Gold standard = triaje de expertos

# Rangos fisiológicos plausibles: limpian el dataset y validan la API.
RANGOS = {
    "edad": (0, 120),
    "temperatura": (30, 45),
    "frecuencia_cardiaca": (20, 250),
    "frecuencia_respiratoria": (4, 80),
    "presion_sistolica": (40, 300),
    "presion_diastolica": (20, 200),
    "saturacion_o2": (50, 100),
    "estado_mental": (0, 3),
    "dolor": (0, 10),
}

# Si el modelo de imagen no supera este AUC (validación cruzada), la API lo ignora:
# una imagen que no discrimina mejor que el azar no debe mover el riesgo del paciente.
AUC_IMAGEN_MIN = 0.60

RANDOM_STATE = 42