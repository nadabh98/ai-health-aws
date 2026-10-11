import pandas as pd
import numpy as np
from pathlib import Path


INPUT = Path("data/processed/who_air_quality_parsed.csv")
OUTPUT = Path("data/processed/health_risk_dataset.csv")


print("Chargement du dataset...")
df = pd.read_csv(INPUT)

print(f"Dataset initial : {df.shape}")

# --------------------------------------------------
# 1. Nettoyage de l'année
# --------------------------------------------------

df["year"] = pd.to_numeric(df["year"], errors="coerce")

before = len(df)

df = df[
    (df["year"] >= 2010) &
    (df["year"] <= 2025)
].copy()

print(f"Lignes supprimées pour année invalide : {before - len(df)}")


# --------------------------------------------------
# 2. Conversion des variables numériques
# --------------------------------------------------

numeric_columns = [
    "pm10_concentration",
    "pm25_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "pm25_tempcov",
    "no2_tempcov",
    "number_stations",
    "latitude",
    "longitude",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# --------------------------------------------------
# 3. Détection des valeurs extrêmes de NO2
# --------------------------------------------------

q1 = df["no2_concentration"].quantile(0.25)
q3 = df["no2_concentration"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 3 * iqr
upper_bound = q3 + 3 * iqr

print("\nNO2 :")
print(f"Q1 : {q1:.2f}")
print(f"Q3 : {q3:.2f}")
print(f"IQR : {iqr:.2f}")
print(f"Seuil inférieur : {lower_bound:.2f}")
print(f"Seuil supérieur : {upper_bound:.2f}")

outliers = (
    (df["no2_concentration"] < lower_bound) |
    (df["no2_concentration"] > upper_bound)
)

print(f"Valeurs NO2 considérées comme extrêmes : {outliers.sum()}")

df.loc[outliers, "no2_concentration"] = np.nan


# --------------------------------------------------
# 4. Création de la variable cible
# --------------------------------------------------

def create_risk_level(pm25):
    if pd.isna(pm25):
        return np.nan
    elif pm25 < 15:
        return "LOW"
    elif pm25 < 35:
        return "MEDIUM"
    else:
        return "HIGH"


df["risk_level"] = df["pm25_concentration"].apply(create_risk_level)

print("\nRépartition initiale des classes :")
print(df["risk_level"].value_counts(dropna=False))


# --------------------------------------------------
# 5. Suppression des lignes sans PM2.5
# --------------------------------------------------

before = len(df)

df = df.dropna(subset=["pm25_concentration", "risk_level"]).copy()

print(
    f"\nLignes supprimées sans PM2.5 / target : "
    f"{before - len(df)}"
)


# --------------------------------------------------
# 6. Variables utilisées pour le Machine Learning
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


# --------------------------------------------------
# 7. Suppression des doublons ML
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=features + ["risk_level"]
).copy()

print(f"Doublons ML supprimés : {before - len(df)}")


# --------------------------------------------------
# 8. Dataset final
# --------------------------------------------------

final_columns = [
    "iso3",
    "city",
    "year",
    "pm10_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "no2_tempcov",
    "number_stations",
    "latitude",
    "longitude",
    "risk_level",
]

df = df[final_columns]


# --------------------------------------------------
# 9. Résultats
# --------------------------------------------------

print("\nRépartition finale des classes :")
print(df["risk_level"].value_counts())

print("\nPourcentage des classes :")
print(
    (df["risk_level"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nValeurs manquantes :")
print(df.isna().sum())

print(f"\nDataset final : {df.shape}")


# --------------------------------------------------
# 10. Sauvegarde
# --------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

print(f"\nDataset sauvegardé : {OUTPUT}")
