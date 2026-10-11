import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from xgboost import XGBClassifier


DATA = "data/processed/health_risk_dataset.csv"

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

LABELS = ["LOW", "MEDIUM", "HIGH"]

LABEL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}


# --------------------------------------------------
# Chargement
# --------------------------------------------------

print("Chargement du dataset...")

df = pd.read_csv(DATA)

X = df[FEATURES]
y = df["risk_level"].map(LABEL_MAP)

groups = (
    df["iso3"].astype(str)
    + "__"
    + df["city"].astype(str)
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf_cm_total = np.zeros((3, 3), dtype=int)
xgb_cm_total = np.zeros((3, 3), dtype=int)


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]


    # --------------------------------------------------
    # Random Forest
    # --------------------------------------------------

    rf = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    print("Random Forest...")

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_cm = confusion_matrix(
        y_test,
        rf_pred,
        labels=[0, 1, 2]
    )

    rf_cm_total += rf_cm


    # --------------------------------------------------
    # XGBoost
    # --------------------------------------------------

    xgb = Pipeline([
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

    print("XGBoost...")

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    xgb_cm = confusion_matrix(
        y_test,
        xgb_pred,
        labels=[0, 1, 2]
    )

    xgb_cm_total += xgb_cm


# --------------------------------------------------
# Fonction d'affichage
# --------------------------------------------------

def display_analysis(name, cm):

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("\nMatrice de confusion cumulée :")
    print()

    print(
        pd.DataFrame(
            cm,
            index=[
                "Réel LOW",
                "Réel MEDIUM",
                "Réel HIGH"
            ],
            columns=[
                "Prédit LOW",
                "Prédit MEDIUM",
                "Prédit HIGH"
            ]
        )
    )


    total = cm.sum()

    correct = np.trace(cm)

    print(f"\nPrédictions correctes : {correct}")
    print(f"Prédictions totales   : {total}")
    print(
        f"Accuracy globale      : "
        f"{correct / total:.4f}"
    )


    # --------------------------------------------------
    # Erreurs détaillées
    # --------------------------------------------------

    print("\nErreurs par transition :")

    errors = []

    for real in range(3):

        for pred in range(3):

            if real != pred:

                count = cm[real, pred]

                errors.append(
                    (
                        LABELS[real],
                        LABELS[pred],
                        int(count)
                    )
                )

    errors.sort(
        key=lambda x: x[2],
        reverse=True
    )

    for real, pred, count in errors:

        print(
            f"{real:6s} -> {pred:6s} : "
            f"{count:5d}"
        )


    # --------------------------------------------------
    # Erreurs graves LOW <-> HIGH
    # --------------------------------------------------

    low_high = cm[0, 2]
    high_low = cm[2, 0]

    print("\nErreurs extrêmes LOW <-> HIGH :")

    print(
        f"LOW -> HIGH : {low_high}"
    )

    print(
        f"HIGH -> LOW : {high_low}"
    )

    print(
        f"Total LOW <-> HIGH : "
        f"{low_high + high_low}"
    )


    # --------------------------------------------------
    # Erreurs adjacentes
    # --------------------------------------------------

    adjacent = (
        cm[0, 1]
        + cm[1, 0]
        + cm[1, 2]
        + cm[2, 1]
    )

    print("\nErreurs entre classes adjacentes :")

    print(
        f"LOW <-> MEDIUM : "
        f"{cm[0,1] + cm[1,0]}"
    )

    print(
        f"MEDIUM <-> HIGH : "
        f"{cm[1,2] + cm[2,1]}"
    )

    print(
        f"Total adjacent : {adjacent}"
    )


    # --------------------------------------------------
    # Pourcentage des erreurs
    # --------------------------------------------------

    total_errors = total - correct

    print("\nRésumé des erreurs :")

    if total_errors > 0:

        print(
            f"Erreurs adjacentes : "
            f"{adjacent / total_errors * 100:.2f}%"
        )

        print(
            f"Erreurs LOW <-> HIGH : "
            f"{(low_high + high_low) / total_errors * 100:.2f}%"
        )


display_analysis(
    "RANDOM FOREST — ANALYSE DES ERREURS",
    rf_cm_total
)

display_analysis(
    "XGBOOST — ANALYSE DES ERREURS",
    xgb_cm_total
)


# --------------------------------------------------
# Comparaison finale
# --------------------------------------------------

print("\n")
print("=" * 70)
print("COMPARAISON DES ERREURS")
print("=" * 70)

print("\nRandom Forest :")
print(rf_cm_total)

print("\nXGBoost :")
print(xgb_cm_total)

print("\nAnalyse terminée.")
