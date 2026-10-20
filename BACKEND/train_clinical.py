"""
Entrenamiento del modelo clínico (sección 3.4.1 de la tesis).

Uso:
    python train_clinical.py                 # usa ../DATASET/clinical_data.csv (datos REALES)
    python train_clinical.py --data ruta.csv
    python train_clinical.py --demo          # datos sintéticos SOLO para probar el pipeline

Flujo: limpieza -> split 70/15/15 estratificado -> compara modelos en validación ->
elige el mejor por ROC-AUC -> calibra -> fija umbrales priorizando Recall ->
evalúa en test -> guarda modelo, métricas, matriz de confusión y tabla comparativa.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FEATURES, RANDOM_STATE, RANGOS, TARGET

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost es opcional
    XGBClassifier = None

BASE = Path(__file__).resolve().parent


def make_demo_data(n=1500):
    """Datos SINTÉTICOS. Solo para probar el código; NUNCA para reportar resultados."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "edad": rng.integers(1, 95, n),
        "sexo": rng.integers(0, 2, n),
        "temperatura": rng.normal(37.2, 0.9, n).round(1),
        "frecuencia_cardiaca": rng.normal(90, 20, n).round(),
        "presion_sistolica": rng.normal(122, 22, n).round(),
        "presion_diastolica": rng.normal(78, 13, n).round(),
        "saturacion_o2": np.clip(rng.normal(95, 3.5, n), 70, 100).round(),
        "sintoma_severidad": rng.integers(0, 3, n),
    })
    z = (0.03 * (df.edad - 50) + 0.04 * (df.frecuencia_cardiaca - 90)
         - 0.25 * (df.saturacion_o2 - 95) + 0.6 * df.sintoma_severidad
         - 0.02 * (df.presion_sistolica - 120) - 1.2 + rng.normal(0, 1, n))
    df[TARGET] = (z > 0).astype(int)
    return df


def clean(df):
    faltan = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltan:
        raise SystemExit(f"Faltan columnas en el CSV: {faltan}")
    n0 = len(df)
    df = df[FEATURES + [TARGET]].drop_duplicates().dropna()
    for col, (lo, hi) in RANGOS.items():
        df = df[df[col].between(lo, hi)]
    df = df[df[TARGET].isin([0, 1])]
    print(f"Limpieza: {n0} -> {len(df)} registros "
          f"({n0 - len(df)} eliminados por duplicados/nulos/fuera de rango)")
    print("Distribución de la clase objetivo:\n", df[TARGET].value_counts(normalize=True).round(3))
    if len(df) < 100 or df[TARGET].nunique() < 2:
        raise SystemExit("Dataset insuficiente para entrenar.")
    return df


def candidatos():
    scale = ("scaler", StandardScaler())
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }
    if XGBClassifier:
        modelos["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           eval_metric="logloss", random_state=RANDOM_STATE)
    return {n: Pipeline([scale, ("clf", m)]) for n, m in modelos.items()}


def metricas(y, proba, umbral=0.5):
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),   # calculado por sklearn: consistente con P y R
        "roc_auc": roc_auc_score(y, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def elegir_umbrales(y_val, p_val, recall_obj=0.90, precision_obj=0.85):
    """Umbral MODERADO: mayor precisión con Recall >= recall_obj (no omitir pacientes en riesgo).
    Umbral GRAVE: menor umbral con Precision >= precision_obj."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= recall_obj
    moderado = float(thr[ok][np.argmax(prec[ok])]) if ok.any() else 0.40
    ok_g = prec >= precision_obj
    grave = float(thr[ok_g].min()) if ok_g.any() else 0.70
    grave = max(grave, moderado + 0.05)
    return round(moderado, 3), round(min(grave, 0.95), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE.parent / "DATASET" / "clinical_data.csv"))
    ap.add_argument("--demo", action="store_true", help="datos sintéticos (solo pruebas)")
    args = ap.parse_args()

    if args.demo:
        print("⚠️  MODO DEMO: datos sintéticos. NO usar estos resultados en la tesis.")
        df, out = make_demo_data(), BASE / "models_demo"
    else:
        path = Path(args.data)
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"No hay dataset real en {path}. Colócalo ahí o usa --demo para probar.")
        df, out = pd.read_csv(path), BASE / "models"
    out.mkdir(exist_ok=True)
    df = clean(df)

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"Split -> train {len(X_tr)} | val {len(X_val)} | test {len(X_te)}")

    # 1) Comparar modelos (la selección se hace SOLO con validación)
    filas = []
    for nombre, pipe in candidatos().items():
        pipe.fit(X_tr, y_tr)
        mv = metricas(y_val, pipe.predict_proba(X_val)[:, 1])
        mt = metricas(y_te, pipe.predict_proba(X_te)[:, 1])
        claves = ("precision", "recall", "f1", "roc_auc")
        filas.append({"modelo": nombre,
                      **{f"val_{k}": mv[k] for k in claves},
                      **{f"test_{k}": mt[k] for k in claves}})
    tabla = pd.DataFrame(filas).sort_values("val_roc_auc", ascending=False)
    tabla.round(4).to_csv(out / "comparacion_modelos.csv", index=False)
    print("\nComparación de modelos (para la Tabla 3):\n", tabla.round(3).to_string(index=False))

    mejor = tabla.iloc[0]["modelo"]
    print(f"\n✔ Modelo elegido (mayor ROC-AUC en validación): {mejor}")

    # 2) Calibrar probabilidades del modelo elegido y fijar umbrales
    final = CalibratedClassifierCV(candidatos()[mejor], method="sigmoid", cv=5)
    final.fit(X_tr, y_tr)
    u_mod, u_grave = elegir_umbrales(y_val, final.predict_proba(X_val)[:, 1])
    print(f"Umbrales (validación): MODERADO >= {u_mod} | GRAVE >= {u_grave}")

    # 3) Evaluación final en test con el umbral MODERADO como punto de alerta
    m_final = metricas(y_te, final.predict_proba(X_te)[:, 1], umbral=u_mod)
    print("\nTEST (modelo final):", {k: round(v, 4) if isinstance(v, float) else v for k, v in m_final.items()})

    # 4) Guardar
    joblib.dump(final, out / "clinical_model.pkl")
    meta = {"modelo": mejor, "features": FEATURES, "umbral_moderado": u_mod, "umbral_grave": u_grave,
            "metricas_test": m_final, "n_train": len(X_tr), "n_val": len(X_val), "n_test": len(X_te),
            "demo": bool(args.demo)}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    try:  # matriz de confusión (Figuras 11/12)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array([[m_final["tn"], m_final["fp"]], [m_final["fn"], m_final["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["Estable", "Riesgo"]); ax.set_yticks([0, 1], ["Estable", "Riesgo"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión - {mejor}")
        fig.tight_layout(); fig.savefig(out / "matriz_confusion.png", dpi=200)
    except ImportError:
        pass
    print(f"\nGuardado en {out}/")


if __name__ == "__main__":
    main()