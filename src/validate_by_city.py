import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

LABEL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}


print("Chargement du dataset...")
df = pd.read_csv(DATA)

X = df[FEATURES]
y = df["risk_level"].map(LABEL_MAP)

# Un groupe = une ville dans un pays
groups = (
    df["iso3"].astype(str)
    + "__"
    + df["city"].astype(str)
)

print(f"Dataset : {df.shape}")
print(f"Nombre de villes/groupes : {groups.nunique()}")

print("\nRépartition globale :")
print(df["risk_level"].value_counts())


# --------------------------------------------------
# Validation croisée par ville
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


rf_scores = []
xgb_scores = []

rf_f1_scores = []
xgb_f1_scores = []

rf_high_f1 = []
xgb_high_f1 = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    print(f"Train : {len(X_train)} lignes")
    print(f"Test  : {len(X_test)} lignes")

    print(
        f"Villes train : {groups_train.nunique()}"
    )

    print(
        f"Villes test  : {groups_test.nunique()}"
    )

    overlap = set(groups_train) & set(groups_test)

    print(f"Villes communes train/test : {len(overlap)}")

    if len(overlap) != 0:
        raise RuntimeError(
            "ERREUR : des villes sont présentes dans train et test."
        )


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


    print("\nEntraînement Random Forest...")

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    rf_accuracy = accuracy_score(
        y_test,
        rf_pred
    )

    rf_macro_f1 = f1_score(
        y_test,
        rf_pred,
        average="macro"
    )

    rf_report = classification_report(
        y_test,
        rf_pred,
        output_dict=True,
        zero_division=0
    )

    rf_high = rf_report["2"]["f1-score"]

    rf_scores.append(rf_accuracy)
    rf_f1_scores.append(rf_macro_f1)
    rf_high_f1.append(rf_high)


    print(
        f"Random Forest | "
        f"Accuracy={rf_accuracy:.4f} | "
        f"Macro-F1={rf_macro_f1:.4f} | "
        f"F1 HIGH={rf_high:.4f}"
    )


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


    print("Entraînement XGBoost...")

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)

    xgb_accuracy = accuracy_score(
        y_test,
        xgb_pred
    )

    xgb_macro_f1 = f1_score(
        y_test,
        xgb_pred,
        average="macro"
    )

    xgb_report = classification_report(
        y_test,
        xgb_pred,
        output_dict=True,
        zero_division=0
    )

    xgb_high = xgb_report["2"]["f1-score"]

    xgb_scores.append(xgb_accuracy)
    xgb_f1_scores.append(xgb_macro_f1)
    xgb_high_f1.append(xgb_high)


    print(
        f"XGBoost      | "
        f"Accuracy={xgb_accuracy:.4f} | "
        f"Macro-F1={xgb_macro_f1:.4f} | "
        f"F1 HIGH={xgb_high:.4f}"
    )


# --------------------------------------------------
# Résultats finaux
# --------------------------------------------------

print("\n")
print("=" * 70)
print("RÉSULTATS VALIDATION PAR VILLE")
print("=" * 70)

print("\nRandom Forest")
print(
    f"Accuracy moyenne : "
    f"{np.mean(rf_scores):.4f} "
    f"+/- {np.std(rf_scores):.4f}"
)

print(
    f"Macro-F1 moyenne : "
    f"{np.mean(rf_f1_scores):.4f} "
    f"+/- {np.std(rf_f1_scores):.4f}"
)

print(
    f"F1 HIGH moyenne : "
    f"{np.mean(rf_high_f1):.4f} "
    f"+/- {np.std(rf_high_f1):.4f}"
)


print("\nXGBoost")
print(
    f"Accuracy moyenne : "
    f"{np.mean(xgb_scores):.4f} "
    f"+/- {np.std(xgb_scores):.4f}"
)

print(
    f"Macro-F1 moyenne : "
    f"{np.mean(xgb_f1_scores):.4f} "
    f"+/- {np.std(xgb_f1_scores):.4f}"
)

print(
    f"F1 HIGH moyenne : "
    f"{np.mean(xgb_high_f1):.4f} "
    f"+/- {np.std(xgb_high_f1):.4f}"
)


print("\nDétail par fold :")

for i in range(5):
    print(
        f"Fold {i+1} | "
        f"RF Accuracy={rf_scores[i]:.4f} | "
        f"RF Macro-F1={rf_f1_scores[i]:.4f} | "
        f"XGB Accuracy={xgb_scores[i]:.4f} | "
        f"XGB Macro-F1={xgb_f1_scores[i]:.4f}"
    )
