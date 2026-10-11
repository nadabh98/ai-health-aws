from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


INPUT_FILE = Path(
    "data/processed/health_risk_dataset.csv"
)

MODEL_FILE = Path(
    "models/random_forest_health_risk.joblib"
)


print("=" * 70)
print("ENTRAINEMENT RANDOM FOREST")
print("=" * 70)


# --------------------------------------------------
# 1. Chargement
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset : {df.shape}")


# --------------------------------------------------
# 2. Features / cible
# --------------------------------------------------

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

X = df[features]
y = df["risk_level"]


print("\nFeatures :")

for feature in features:
    print(f"- {feature}")


print("\nCible : risk_level")

print(
    y.value_counts()
)


# --------------------------------------------------
# 3. Train / Test
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print("\nTrain :", X_train.shape)
print("Test  :", X_test.shape)


# --------------------------------------------------
# 4. Pipeline
# --------------------------------------------------

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    ),
])


# --------------------------------------------------
# 5. Entraînement
# --------------------------------------------------

print("\nEntraînement...")

pipeline.fit(
    X_train,
    y_train,
)


print("Entraînement terminé.")


# --------------------------------------------------
# 6. Prédictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 7. Évaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
)


print("\n" + "=" * 70)
print("RESULTATS")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Macro-F1 : {macro_f1:.4f}"
)


print("\nClassification report :")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)


print("\nMatrice de confusion :")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )
)


# --------------------------------------------------
# 8. Sauvegarde
# --------------------------------------------------

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    pipeline,
    MODEL_FILE,
)


print("\n" + "=" * 70)
print("MODELE SAUVEGARDE")
print("=" * 70)

print(
    f"\nFichier : {MODEL_FILE}"
)

print("\nTerminé.")
