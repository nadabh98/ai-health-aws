import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


DATA = "data/processed/health_risk_dataset.csv"
MODEL_PATH = "models/xgboost_health_risk_final.joblib"


FEATURES = [
    "year",
    "pm10_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "no2_tempcov",
    "number_stations",
    "latitude",
    "longitude",
]


LABEL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}


print("Chargement du dataset...")

df = pd.read_csv(DATA)

X = df[FEATURES]
y = df["risk_level"].map(LABEL_MAP)


print(f"Dataset : {X.shape}")

print("\nRépartition des classes :")
print(df["risk_level"].value_counts())


model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
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


print("\nEntraînement du modèle final...")

model.fit(X, y)


joblib.dump(model, MODEL_PATH)


print("\nModèle final sauvegardé :")
print(MODEL_PATH)
