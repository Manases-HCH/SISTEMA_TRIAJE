"""
Convierte el CSV de Kaggle "Emergency Service - Triage Application" (KTAS)
al formato de train_clinical.py  ->  DATASET/clinical_data.csv

Uso:  python prepare_dataset.py ..\DATASET\ktas_raw.csv
"""
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE.parent / "DATASET" / "ktas_raw.csv"
OUT = BASE.parent / "DATASET" / "clinical_data.csv"

df = None
for enc in ("utf-8", "latin-1"):
    try:
        df = pd.read_csv(SRC, sep=None, engine="python", encoding=enc)
        break
    except UnicodeDecodeError:
        continue
df.columns = [c.strip() for c in df.columns]

necesarias = ["Sex", "Age", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation", "Mental", "KTAS_expert"]
faltan = [c for c in necesarias if c not in df.columns]
if faltan:
    raise SystemExit(f"Columnas no encontradas: {faltan}\nColumnas del CSV: {list(df.columns)}")


def num(s):  # '??', vacíos y similares pasan a NaN
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


dolor = num(df["NRS_pain"])
if "Pain" in df.columns:               # Pain = 0 -> sin dolor -> NRS = 0
    dolor = dolor.where(num(df["Pain"]) != 0, 0)
ktas = num(df["KTAS_expert"])

out = pd.DataFrame({
    "edad": num(df["Age"]),
    "sexo": num(df["Sex"]).map({2: 1, 1: 0}),      # original: 1 = mujer, 2 = hombre
    "temperatura": num(df["BT"]),
    "frecuencia_cardiaca": num(df["HR"]),
    "frecuencia_respiratoria": num(df["RR"]),
    "presion_sistolica": num(df["SBP"]),
    "presion_diastolica": num(df["DBP"]),
    "saturacion_o2": num(df["Saturation"]),
    "estado_mental": num(df["Mental"]) - 1,        # original: 1 alerta ... 4 inconsciente
    "dolor": dolor,
    "riesgo": (ktas <= 3).astype("Int64").where(ktas.notna()),  # KTAS 1-3 = riesgo
})
n0 = len(out)
# NO se descartan filas por signos vitales faltantes: se imputan dentro del pipeline.
out = out.dropna(subset=["riesgo", "edad", "sexo"])
out["riesgo"] = out["riesgo"].astype(int)
out.to_csv(OUT, index=False)

print(f"{n0} filas -> {len(out)} usables. Guardado en {OUT}")
print("\nDatos faltantes (%):\n", (out.drop(columns="riesgo").isna().mean() * 100).round(1).to_string())
print("\nDistribución de riesgo (0 = KTAS 4-5, 1 = KTAS 1-3):")
print(out["riesgo"].value_counts(normalize=True).round(3).to_string())