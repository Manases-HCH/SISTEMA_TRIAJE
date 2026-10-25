"""
Modelo clínico (sección 3.4.1).  Uso:
    python train_clinical.py          # datos REALES en ../DATASET/clinical_data.csv
    python train_clinical.py --demo   # datos sintéticos, SOLO para probar el código

Flujo: limpieza -> split 80/20 estratificado (train/test) ->
en TRAIN: validación cruzada de 5 pliegues + búsqueda de hiperparámetros para 4 modelos ->
elige el mejor por ROC-AUC de validación -> calibra -> umbrales por Recall (con predicciones
fuera de pliegue) -> evalúa UNA vez en TEST (con IC95% del AUC por bootstrap).
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import FEATURES, RANDOM_STATE, RANGOS, TARGET

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

BASE = Path(__file__).resolve().parent


def make_demo_data(n=1200):
    """SINTÉTICO. Solo para probar el código; NUNCA para reportar resultados."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "edad": rng.integers(1, 95, n), "sexo": rng.integers(0, 2, n),
        "temperatura": rng.normal(36.8, 0.8, n).round(1),
        "frecuencia_cardiaca": rng.normal(88, 20, n).round(),
        "frecuencia_respiratoria": rng.normal(20, 4, n).round(),
        "presion_sistolica": rng.normal(130, 25, n).round(),
        "presion_diastolica": rng.normal(80, 14, n).round(),
        "saturacion_o2": np.clip(rng.normal(96, 3, n), 70, 100).round(),
        "estado_mental": rng.choice([0, 1, 2, 3], n, p=[.88, .07, .04, .01]),
        "dolor": rng.integers(0, 11, n),
    })
    z = (0.03 * (df.edad - 50) + 0.03 * (df.frecuencia_cardiaca - 88) + 0.08 * (df.frecuencia_respiratoria - 20)
         - 0.25 * (df.saturacion_o2 - 96) + 0.8 * df.estado_mental + 0.1 * df.dolor
         - 0.015 * (df.presion_sistolica - 130) + rng.normal(0, 1.2, n))
    df[TARGET] = (z > 0.4).astype(int)
    df.loc[rng.random(n) < 0.15, "saturacion_o2"] = np.nan   # simula faltantes
    return df


def clean(df):
    faltan = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltan:
        raise SystemExit(f"Faltan columnas en el CSV: {faltan}")
    n0 = len(df)
    df = df[FEATURES + [TARGET]].drop_duplicates().dropna(subset=[TARGET])
    for col, (lo, hi) in RANGOS.items():          # valores imposibles -> NaN (se imputan), no se borra la fila
        df.loc[~df[col].between(lo, hi) & df[col].notna(), col] = np.nan
    df = df[df[TARGET].isin([0, 1])]
    print(f"Limpieza: {n0} -> {len(df)} registros")
    print("Datos faltantes (%):\n", (df[FEATURES].isna().mean() * 100).round(1).to_string())
    print("Distribución de la clase objetivo:\n", df[TARGET].value_counts(normalize=True).round(3).to_string())
    if len(df) < 200 or df[TARGET].nunique() < 2:
        raise SystemExit("Dataset insuficiente para entrenar.")
    return df


def candidatos():
    def pipe(clf):
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scaler", StandardScaler()), ("clf", clf)])
    c = {
        "Regresión Logística": (pipe(LogisticRegression(max_iter=3000, class_weight="balanced")),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                      random_state=RANDOM_STATE, n_jobs=-1)),
                          {"clf__max_depth": [3, 5, 8, None], "clf__min_samples_leaf": [1, 5, 10]}),
        "SVM": (pipe(CalibratedClassifierCV(SVC(class_weight="balanced"), cv=3, ensemble=False)),
                {"clf__estimator__C": [0.1, 1, 10], "clf__estimator__gamma": ["scale", 0.01, 0.1]}),
    }
    if XGBClassifier:
        c["XGBoost"] = (pipe(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
                        {"clf__max_depth": [2, 3, 4], "clf__n_estimators": [100, 300],
                         "clf__learning_rate": [0.03, 0.1]})
    return c


def metricas(y, proba, umbral=0.5):
    y = np.asarray(y)
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def ic_auc(y, proba, n=1000):
    rng, y, proba, aucs = np.random.default_rng(RANDOM_STATE), np.asarray(y), np.asarray(proba), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], proba[i]))
    return [round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)]


def elegir_umbrales(y, p, recall_obj=0.90, precision_obj=0.85):
    """MODERADO: mayor precisión con Recall >= 0.90 (no omitir pacientes en riesgo).
    GRAVE: menor umbral con Precisión >= 0.85."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = rec >= recall_obj
    mod = float(thr[ok][np.argmax(prec[ok])]) if ok.any() else 0.40
    okg = prec >= precision_obj
    grave = float(thr[okg].min()) if okg.any() else 0.70
    grave = max(grave, mod + 0.05)
    return round(mod, 3), round(min(grave, 0.95), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE.parent / "DATASET" / "clinical_data.csv"))
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        print("⚠️  MODO DEMO: datos sintéticos. NO usar estos resultados en la tesis.")
        df, out = make_demo_data(), BASE / "models_demo"
    else:
        path = Path(args.data)
        if not path.exists() or path.stat().st_size == 0:
            raise SystemExit(f"No hay dataset en {path}. Ejecuta prepare_dataset.py primero.")
        df, out = pd.read_csv(path), BASE / "models"
    out.mkdir(exist_ok=True)
    df = clean(df)

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"\nSplit -> train {len(X_tr)} | test {len(X_te)}  (validación = 5 pliegues dentro de train)")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    filas, mejores = [], {}
    for nombre, (pipe, grid) in candidatos().items():
        print(f"  Ajustando {nombre}...")
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1).fit(X_tr, y_tr)
        mejores[nombre] = gs.best_estimator_
        oof = cross_val_predict(gs.best_estimator_, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        mv, mt = metricas(y_tr, oof), metricas(y_te, gs.best_estimator_.predict_proba(X_te)[:, 1])
        filas.append({"modelo": nombre, "mejores_params": str(gs.best_params_),
                      "val_auc": gs.best_score_, **{f"val_{k}": mv[k] for k in ("precision", "recall", "f1")},
                      **{f"test_{k}": mt[k] for k in ("precision", "recall", "f1", "roc_auc")}})
    tabla = pd.DataFrame(filas).sort_values("val_auc", ascending=False)
    tabla.drop(columns="mejores_params").round(4).to_csv(out / "comparacion_modelos.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nComparación (val = validación cruzada en train; test = hold-out):\n",
          tabla.drop(columns="mejores_params").round(3).to_string(index=False))

    mejor = tabla.iloc[0]["modelo"]
    print(f"\n✔ Modelo elegido (mayor ROC-AUC en validación): {mejor}  {tabla.iloc[0]['mejores_params']}")

    # Calibración + umbrales con predicciones fuera de pliegue
    final = CalibratedClassifierCV(mejores[mejor], method="sigmoid", cv=5)
    oof_cal = cross_val_predict(final, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    u_mod, u_grave = elegir_umbrales(y_tr, oof_cal)
    final.fit(X_tr, y_tr)
    print(f"Umbrales: MODERADO >= {u_mod} | GRAVE >= {u_grave}")

    p_te = final.predict_proba(X_te)[:, 1]
    m_alerta, m_05 = metricas(y_te, p_te, u_mod), metricas(y_te, p_te, 0.5)
    ic = ic_auc(y_te, p_te)
    print(f"\nTEST  AUC = {m_05['roc_auc']:.3f}  IC95% {ic}")
    print("TEST  umbral 0.5      :", {k: round(v, 3) for k, v in m_05.items() if k in ("precision", "recall", "f1")})
    print("TEST  umbral alerta   :", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m_alerta.items()})

    joblib.dump(final, out / "clinical_model.pkl")
    meta = {"modelo": mejor, "features": FEATURES, "umbral_moderado": u_mod, "umbral_grave": u_grave,
            "auc_validacion": round(float(tabla.iloc[0]["val_auc"]), 4), "auc_test": round(m_05["roc_auc"], 4),
            "auc_test_ic95": ic, "metricas_test_umbral_alerta": m_alerta, "metricas_test_umbral_0.5": m_05,
            "n_train": len(X_tr), "n_test": len(X_te), "demo": bool(args.demo)}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array([[m_alerta["tn"], m_alerta["fp"]], [m_alerta["fn"], m_alerta["tp"]]])
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