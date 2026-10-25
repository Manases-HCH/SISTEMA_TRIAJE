import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import RANDOM_STATE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = (224, 224)
EPOCHS = 15
# Sin internet o en pruebas: VISION_NO_PRETRAIN=1 (pesos aleatorios; solo para probar el código)
WEIGHTS = None if os.getenv("VISION_NO_PRETRAIN") else "imagenet"


def cargar(carpeta):
    X, y = [], []
    for etiqueta, nombre in enumerate(("estable", "riesgo")):
        archivos = [p for p in (carpeta / nombre).glob("*") if p.suffix.lower() in EXT]
        for p in archivos:
            X.append(np.asarray(Image.open(p).convert("RGB").resize(SIZE), dtype="float32"))  # 0-255
            y.append(etiqueta)
        print(f"  {nombre}: {len(archivos)} imágenes")
    return np.stack(X), np.array(y)


def build_model():
    base = keras.applications.MobileNetV2(input_shape=SIZE + (3,), include_top=False, weights=WEIGHTS)
    base.trainable = False                      # solo se entrena la cabeza (pocas imágenes)
    inp = layers.Input(SIZE + (3,))
    x = layers.RandomFlip("horizontal")(inp)    # aumento de datos: activo solo al entrenar
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.15)(x)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)   # preprocesado de MobileNetV2 DENTRO del modelo
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def entrenar(X, y):
    pesos = {0: len(y) / (2 * (y == 0).sum()), 1: len(y) / (2 * (y == 1).sum())}
    m = build_model()
    m.fit(X, y, epochs=EPOCHS, batch_size=8, class_weight=pesos, verbose=0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    keras.utils.set_random_seed(RANDOM_STATE)

    if args.demo:
        print("⚠️  MODO DEMO: imágenes sintéticas. NO usar estos resultados en la tesis.")
        rng = np.random.default_rng(RANDOM_STATE)
        y = np.array([0] * 20 + [1] * 20)
        X = rng.integers(0, 256, (40, *SIZE, 3)).astype("float32")
        X[y == 1] *= 0.7
        out = BASE / "models_demo"
    else:
        carpeta = BASE.parent / "DATASET" / "images"
        print(f"Cargando imágenes de {carpeta}")
        if not carpeta.exists():
            raise SystemExit(f"No existe {carpeta}. Crea las carpetas 'estable' y 'riesgo' con tus imágenes.")
        X, y = cargar(carpeta)
        out = BASE / "models"
        if min((y == 0).sum(), (y == 1).sum()) < 10:
            raise SystemExit("Se necesitan al menos 10 imágenes por clase.")
    out.mkdir(exist_ok=True)

    n_pliegues = int(min(5, (y == 0).sum(), (y == 1).sum()))
    oof = np.zeros(len(y))
    for i, (tr, va) in enumerate(StratifiedKFold(n_pliegues, shuffle=True, random_state=RANDOM_STATE).split(X, y), 1):
        print(f"  Pliegue {i}/{n_pliegues}...")
        oof[va] = entrenar(X[tr], y[tr]).predict(X[va], verbose=0).ravel()
        keras.backend.clear_session()

    pred = (oof >= 0.5).astype(int)
    m = {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
         "f1": f1_score(y, pred, zero_division=0), "auc_cv": roc_auc_score(y, oof)}
    print("\nMétricas (validación cruzada, todas las imágenes):", {k: round(v, 3) for k, v in m.items()})
    if m["auc_cv"] < 0.60:
        print("⚠️  AUC < 0.60: el modelo de imagen no discrimina mejor que el azar. "
              "La API NO lo usará hasta que consigas más/mejores imágenes.")

    entrenar(X, y).save(out / "vision_model.keras")      # modelo final con todas las imágenes
    meta = {"arquitectura": "MobileNetV2 (transfer learning)", "n_imagenes": int(len(y)),
            "n_estable": int((y == 0).sum()), "n_riesgo": int((y == 1).sum()), "pliegues": n_pliegues,
            **{k: round(float(v), 4) for k, v in m.items()}, "demo": bool(args.demo)}
    (out / "vision_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Guardado en {out}/")


if __name__ == "__main__":
    main()