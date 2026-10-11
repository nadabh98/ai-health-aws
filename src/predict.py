import pandas as pd
import joblib


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


LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}


print("Chargement du modèle...")

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Exemple de données
# --------------------------------------------------

data = pd.DataFrame([{
    "year": 2025,
    "pm10_concentration": 25.0,
    "no2_concentration": 18.0,
    "pm10_tempcov": 0.90,
    "no2_tempcov": 0.90,
    "number_stations": 3,
    "latitude": -3.10,
    "longitude": -60.10,
}])


# --------------------------------------------------
# Prédiction
# --------------------------------------------------

prediction = model.predict(data[FEATURES])[0]

probabilities = model.predict_proba(
    data[FEATURES]
)[0]


risk = LABELS[int(prediction)]


print("\n" + "=" * 50)
print("PRÉDICTION DU RISQUE SANITAIRE ENVIRONNEMENTAL")
print("=" * 50)

print(f"\nNiveau prédit : {risk}")

print("\nProbabilités :")

for class_id, probability in enumerate(probabilities):

    print(
        f"{LABELS[class_id]:8s} : "
        f"{probability * 100:.2f}%"
    )


print("\nVariables utilisées :")

for feature in FEATURES:

    print(
        f"{feature:22s} : "
        f"{data.iloc[0][feature]}"
    )
