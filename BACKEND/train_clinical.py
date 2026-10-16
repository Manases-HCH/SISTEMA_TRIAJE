import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Simulación o carga de tus registros clínicos
# Variables: [edad, sexo_binario, temp, fc, pas, pad, sintoma_cod]
# Target: 0 (Estable), 1 (Riesgo/Grave)
np.random.seed(42)
X_dummy = np.random.rand(1000, 7)
y_dummy = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_dummy, y_dummy)

# Guardar el modelo en la carpeta models/
joblib.dump(pipeline, 'models/clinical_model.pkl')
print("Modelo clínico guardado exitosamente.")