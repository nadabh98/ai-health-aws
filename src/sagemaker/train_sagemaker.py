import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib


# =========================
# Configuration
# =========================

DATA_PATH = os.environ.get(
    "SM_CHANNEL_TRAIN",
    "data/processed"
)

MODEL_DIR = os.environ.get(
    "SM_MODEL_DIR",
    "models/sagemaker"
)


# =========================
# Chargement des données
# =========================

csv_files = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("Aucun fichier CSV trouvé dans le canal train.")

data = pd.read_csv(csv_files[0])

print(f"Dataset chargé : {data.shape}")


# =========================
# Features / Target
# =========================

features = [
    "year",
    "pm10_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "no2_tempcov",
    "number_stations",
    "latitude",
    "longitude",
]

target = "risk_level"

X = data[features]
y = data[target].map({
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
})


# =========================
# Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")


# =========================
# Pipeline ML
# =========================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "classifier",
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
    )
])


# =========================
# Entraînement
# =========================

print("Entraînement du modèle SageMaker...")

model.fit(X_train, y_train)

print("Entraînement terminé.")


# =========================
# Sauvegarde
# =========================

os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(
    MODEL_DIR,
    "model.joblib"
)

joblib.dump(model, model_path)

print(f"Modèle sauvegardé : {model_path}")
