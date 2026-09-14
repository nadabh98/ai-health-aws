# AI Health Risk Prediction — AWS Cloud Platform

## ODD 3 — Bonne santé et bien-être

Projet d'intelligence artificielle et de cloud computing visant
à prédire un niveau de risque sanitaire environnemental régional
à partir de données publiques sur la qualité de l'air.

Le projet ne réalise pas de diagnostic médical individuel.
Il s'agit d'un modèle de prédiction de risque environnemental
à l'échelle de villes/régions.

## Objectifs

- Collecter des données publiques de qualité de l'air
- Nettoyer et analyser les données
- Construire un modèle de Machine Learning
- Évaluer ses performances
- Stocker les données et le modèle sur Amazon S3
- Expérimenter un déploiement avec Amazon SageMaker
- Exposer une prédiction via AWS Lambda
- Créer une API avec API Gateway
- Développer un dashboard Streamlit
- Superviser l'application avec CloudWatch

## Technologies

### Data & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib

### AWS

- Amazon S3
- Amazon SageMaker
- AWS Lambda
- Amazon API Gateway
- Amazon CloudWatch
- AWS IAM

### Application

- Streamlit
- REST API
- Docker

## Architecture

Données publiques
        ↓
Python / Pandas
        ↓
Machine Learning
        ↓
Amazon S3
        ↓
Amazon SageMaker
        ↓
AWS Lambda
        ↓
API Gateway
        ↓
Dashboard Streamlit

## Dataset

WHO Ambient Air Quality Database.

Les données utilisées sont agrégées à l'échelle des villes
et ne contiennent pas de données médicales individuelles.

## Budget

Le projet est conçu pour rester dans un budget AWS très limité.

Les ressources payantes sont créées uniquement lorsque nécessaire
pour les tests et supprimées ou arrêtées après utilisation.

## Statut

Projet en cours de développement.
