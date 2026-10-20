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

necesarias = ["Sex", "Age", "NRS_pain", "SBP", "DBP", "HR", "BT", "Saturation", "KTAS_expert"]
faltan = [c for c in necesarias if c not in df.columns]
if faltan:
    raise SystemExit(f"Columnas no encontradas: {faltan}\nColumnas del CSV: {list(df.columns)}")


def num(s):  # '??', vacíos y similares pasan a NaN
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.strip(), errors="coerce")


dolor = num(df["NRS_pain"]).fillna(0)  # sin dolor registrado = 0
out = pd.DataFrame({
    "edad": num(df["Age"]),
    "sexo": num(df["Sex"]).map({2: 1, 1: 0}),   # en el original: 1 = mujer, 2 = hombre
    "temperatura": num(df["BT"]),
    "frecuencia_cardiaca": num(df["HR"]),
    "presion_sistolica": num(df["SBP"]),
    "presion_diastolica": num(df["DBP"]),
    "saturacion_o2": num(df["Saturation"]),
    # severidad = dolor NRS categorizado: 0-3 leve, 4-6 moderado, 7-10 severo
    "sintoma_severidad": pd.cut(dolor, [-1, 3, 6, 10], labels=[0, 1, 2]).astype("Int64"),
    # gold standard: KTAS de 3 expertos. Niveles 1-3 = riesgo / urgente
    "riesgo": (num(df["KTAS_expert"]) <= 3).astype("Int64").where(num(df["KTAS_expert"]).notna()),
})
n0 = len(out)
out = out.dropna()
out.astype({"sintoma_severidad": int, "riesgo": int}).to_csv(OUT, index=False)

print(f"{n0} filas -> {len(out)} usables. Guardado en {OUT}")
print("Distribución de riesgo (0=KTAS 4-5, 1=KTAS 1-3):")
print(out["riesgo"].value_counts(normalize=True).round(3).to_string())