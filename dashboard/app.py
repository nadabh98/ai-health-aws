import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Health Risk Prediction",
    page_icon="🌍",
    layout="wide"
)

API_URL = os.getenv("API_URL", "").rstrip("/")

st.title("🌍 AI Health Risk Prediction")
st.caption("Estimation du niveau de risque environnemental à partir des données de pollution.")

st.info(
    "Cette application est une démonstration technique. "
    "Elle estime une classe de risque à partir des données fournies "
    "et ne constitue pas un diagnostic médical."
)

if not API_URL:
    st.error(
        "L'URL de l'API est absente. "
        "Configurez la variable d'environnement API_URL avant de lancer l'application."
    )
    st.stop()

st.subheader("1. Données environnementales")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Année",
        min_value=2000,
        max_value=2035,
        value=2024,
        step=1
    )
    pm10 = st.number_input(
        "Concentration PM10",
        min_value=0.0,
        max_value=1000.0,
        value=20.0,
        step=1.0
    )
    no2 = st.number_input(
        "Concentration NO₂",
        min_value=0.0,
        max_value=1000.0,
        value=15.0,
        step=1.0
    )
    stations = st.number_input(
        "Nombre de stations de mesure",
        min_value=1,
        max_value=10000,
        value=5,
        step=1
    )

with col2:
    pm10_coverage = st.slider(
        "Couverture des mesures PM10",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    no2_coverage = st.slider(
        "Couverture des mesures NO₂",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=48.8566,
        format="%.4f"
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=2.3522,
        format="%.4f"
    )

if st.button("Calculer le niveau de risque", type="primary"):
    payload = {
        "year": int(year),
        "pm10_concentration": float(pm10),
        "no2_concentration": float(no2),
        "pm10_tempcov": float(pm10_coverage),
        "no2_tempcov": float(no2_coverage),
        "number_stations": int(stations),
        "latitude": float(latitude),
        "longitude": float(longitude)
    }

    with st.spinner("Interrogation du modèle AWS en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            # Compatible avec une réponse Lambda proxy
            if isinstance(result.get("body"), str):
                import json
                result = json.loads(result["body"])

            risk = result.get("risk_level")
            probabilities = result.get("probabilities", {})

            if risk not in {"LOW", "MEDIUM", "HIGH"}:
                st.error("La réponse de l'API ne contient pas de niveau de risque valide.")
                st.json(result)
            else:
                st.subheader("2. Résultat de la prédiction")

                labels = {
                    "LOW": "Faible",
                    "MEDIUM": "Modéré",
                    "HIGH": "Élevé"
                }

                st.metric(
                    "Niveau de risque estimé",
                    labels[risk],
                    help="Classe prédite par le modèle XGBoost."
                )

                st.write("**Probabilités fournies par le modèle**")

                for key in ("LOW", "MEDIUM", "HIGH"):
                    probability = float(probabilities.get(key, 0))
                    st.write(f"{labels[key]} : {probability * 100:.2f} %")
                    st.progress(max(0.0, min(1.0, probability)))

                st.caption(
                    "Les probabilités décrivent la sortie du modèle. "
                    "Elles ne garantissent pas le risque réel."
                )

        except requests.exceptions.Timeout:
            st.error("L'API n'a pas répondu dans le délai prévu.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"L'API a renvoyé une erreur HTTP : {exc}")
            st.code(response.text[:1000])
        except requests.exceptions.RequestException as exc:
            st.error(f"Impossible de joindre l'API : {exc}")
        except (ValueError, TypeError) as exc:
            st.error(f"Réponse de l'API impossible à interpréter : {exc}")

st.divider()
st.caption("Projet de démonstration — Streamlit · AWS API Gateway · AWS Lambda · XGBoost")
