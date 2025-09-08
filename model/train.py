import pathlib
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1) Cargar dataset de stroke
csv_path = pathlib.Path("data/healthcare-dataset-stroke-data.csv")
df = pd.read_csv(csv_path)

# 2) Limpieza mínima: convertir 'N/A' de bmi a NaN y a float
df["bmi"] = df["bmi"].replace("N/A", np.nan).astype(float)

# 3) Separar features y target
y = df["stroke"].astype(int)
X = df.drop(columns=["stroke", "id"])  # 'id' no aporta al modelo

# 4) Definir columnas por tipo
numeric_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
categorical_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# 5) Preprocesamiento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 6) Modelo (class_weight para desbalanceo)
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=0,
    n_jobs=-1,
    class_weight="balanced"
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])

# 7) Split + entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

print("Training model...")
pipe.fit(X_train, y_train)

# 8) Métricas + umbral óptimo por F1 en validación
from sklearn.metrics import precision_recall_curve, f1_score, classification_report

y_proba = pipe.predict_proba(X_test)[:, 1]

# Curva P-R y búsqueda de threshold que maximiza F1
prec, rec, th = precision_recall_curve(y_test, y_proba)
# precision_recall_curve devuelve thresholds de tamaño len(prec)-1
f1s = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-15)
best_idx = f1s.argmax()
best_threshold = float(th[best_idx])

print("Evaluation on holdout (default threshold=0.5):")
print(classification_report(y_test, (y_proba >= 0.5).astype(int), digits=4))

print(f"\nBest F1 threshold found: {best_threshold:.3f}")
print("Evaluation on holdout (best F1 threshold):")
print(classification_report(y_test, (y_proba >= best_threshold).astype(int), digits=4))


FEATURES = X.columns.tolist()
# 9) Guardar modelo + threshold
model_path = pathlib.Path("model/stroke-model-v1.joblib")
print(f"Saving model + threshold to {model_path} ...")
dump({"pipe": pipe, "threshold": best_threshold, "features": FEATURES}, model_path)
print("Done.")