# AI Health Risk Prediction — AWS Cloud Platform

An end-to-end Machine Learning and cloud engineering project that predicts
regional environmental health risk levels from public air-quality data.

**Domain:** Environmental Data Science · Machine Learning · AWS Cloud  
**UN Sustainable Development Goal:** SDG 3 — Good Health and Well-being

## Overview

Air pollution is an important environmental risk factor for public health.
This project explores how public air-quality data can be processed and used
to classify environmental risk into three categories: `LOW`, `MEDIUM`, and
`HIGH`.

The project combines data preparation, model training, model serialization,
containerization, a serverless prediction API, and an interactive dashboard.

**Important:** This application estimates environmental risk from aggregated
air-quality indicators. It does not diagnose medical conditions, predict an
individual's health outcome, or replace professional medical advice.

## Project objectives

- Collect and process public air-quality data.
- Clean, transform, and validate environmental datasets.
- Train and compare Machine Learning models.
- Build a final XGBoost classification model.
- Store datasets and model artifacts in Amazon S3.
- Package the inference application in a Docker container.
- Deploy predictions through AWS Lambda.
- Expose the prediction service through Amazon API Gateway.
- Build an interactive Streamlit dashboard.
- Use AWS logs and service-level monitoring tools to investigate issues.
- Keep cloud resource usage and costs under control.

## Dataset

The project uses the **WHO Ambient Air Quality Database**, a public source
of air-quality measurements and related metadata.

The data preparation workflow includes parsing, cleaning, transformation,
and the creation of a model-ready dataset.

### Data preparation

The initial parsing workflow processed a source table of approximately
55,221 rows and 13 columns into a structured dataset of approximately
45,738 rows and 21 columns.

The processed data covers multiple countries and cities, with observations
reported across the 2010–2025 period in the source data used by the project.

The pipeline includes scripts for loading the source data, cleaning records,
preparing features, training models, analysing errors, and validating results
by city.

### Model input features

The deployed inference function uses eight input features:

| Feature | Description |
|---|---|
| `year` | Observation year |
| `pm10_concentration` | PM10 concentration indicator |
| `no2_concentration` | NO₂ concentration indicator |
| `pm10_tempcov` | PM10 temporal coverage indicator |
| `no2_tempcov` | NO₂ temporal coverage indicator |
| `number_stations` | Number of measurement stations |
| `latitude` | Geographic latitude |
| `longitude` | Geographic longitude |

The model returns one of three environmental risk categories:
`LOW`, `MEDIUM`, or `HIGH`.

The dataset contains aggregated environmental information, not individual
medical records.

## Machine Learning

The repository contains scripts for data preparation, training, prediction,
error analysis, and geographic validation.

### Models and tools

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Joblib

The final inference artifact is named:

`xgboost_health_risk_final.joblib`

Model artifacts and datasets are excluded from Git to avoid publishing
large binary files. The final model is stored separately in Amazon S3.

### Evaluation and validation

The project includes dedicated scripts for:

- Training baseline and XGBoost models.
- Analysing classification errors.
- Comparing model predictions.
- Validating performance by city.

Evaluation metrics should be interpreted alongside the class distribution,
data coverage, and geographic differences. A predicted risk category is
not a causal explanation of pollution-related health outcomes.

## AWS architecture

The deployed prediction path is:

```text
Public WHO air-quality data
           |
           v
Python data preparation
           |
           v
Machine Learning training
           |
           v
Model artifact in Amazon S3
           |
           v
Docker container stored in Amazon ECR
           |
           v
AWS Lambda inference function
           |
           v
Amazon API Gateway — POST /predict
           |
           v
Streamlit dashboard
```

### AWS services

| Service | Role |
|---|---|
| Amazon S3 | Storage for datasets and model artifacts |
| Amazon ECR | Storage for the Lambda container image |
| AWS Lambda | Serverless model inference |
| Amazon API Gateway | HTTP endpoint for prediction requests |
| AWS IAM | Access control and execution permissions |
| Amazon CloudWatch | Lambda logs and troubleshooting |
| Amazon SageMaker | Training and experimentation work |

**Deployment note:** The SageMaker Studio environment used during
experimentation has been deleted to avoid unnecessary ongoing costs.
The prediction service uses the model artifact packaged for Lambda; the
active inference path does not require a running SageMaker Studio domain.

## Prediction API

The deployed HTTP API exposes the following route:

`POST /predict`

API base URL:

`https://fwws908e3b.execute-api.eu-west-3.amazonaws.com`

Example request body:

```json
{
  "year": 2024,
  "pm10_concentration": 25.0,
  "no2_concentration": 15.0,
  "pm10_tempcov": 0.8,
  "no2_tempcov": 0.8,
  "number_stations": 5,
  "latitude": 48.8566,
  "longitude": 2.3522
}
```

Example response structure:

```json
{
  "risk_level": "LOW",
  "probabilities": {
    "LOW": 0.9389,
    "MEDIUM": 0.0602,
    "HIGH": 0.0009
  }
}
```

The response above is an example of a successful test, not a guarantee
of the prediction for every input. The returned probabilities depend on
the input values and the trained model.

## Streamlit dashboard

The interactive dashboard is available in `dashboard/app.py`.

It allows users to enter environmental indicators, submit a request to the
prediction API, and view the returned risk category and probabilities.

The dashboard communicates with the API using the `API_URL` environment
variable.

### Run the dashboard locally

From the repository root, activate your Python environment and install
the dashboard dependencies:

```bash
pip install -r dashboard/requirements.txt
```

Set the API URL:

```bash
export API_URL="https://fwws908e3b.execute-api.eu-west-3.amazonaws.com"
```

Start Streamlit:

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal.

The dashboard code is included in the repository. A public Streamlit
hosting deployment is not assumed by these instructions.

## Repository structure

```text
ai-health-aws/
├── dashboard/
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── analyze_errors.py
│   ├── clean_data.py
│   ├── load_who_data.py
│   ├── predict.py
│   ├── train.py
│   ├── train_final.py
│   ├── train_xgboost.py
│   ├── validate_by_city.py
│   ├── lambda/
│   │   ├── Dockerfile
│   │   ├── lambda_function.py
│   │   └── requirements.txt
│   └── sagemaker/
│       └── train_sagemaker.py
├── .gitignore
├── README.md
└── requirements.txt
```

Raw datasets, processed data files, and serialized model artifacts are
excluded from Git through `.gitignore`. The corresponding directories
are retained using `.gitkeep` files.

## Technology stack

**Programming and data:** Python, Pandas, NumPy

**Machine Learning:** Scikit-learn, XGBoost, Joblib

**Cloud and deployment:** AWS Lambda, Amazon S3, Amazon ECR,
Amazon API Gateway, AWS IAM, Amazon CloudWatch, Docker

**User interface:** Streamlit

**Version control:** Git and GitHub

## Cost management

Cloud resources can generate charges even when they are not actively used.
The project therefore follows a cost-conscious approach:

- Remove experimental resources when they are no longer needed.
- Avoid leaving interactive compute environments running.
- Review Lambda, ECR, API Gateway, S3, and CloudWatch usage.
- Check AWS Billing and Cost Management regularly.
- Set budget alerts where available.

Deleting SageMaker Studio does not automatically remove every other AWS
resource or erase charges already incurred. Billing data may also take
time to update.

## Limitations and responsible use

- Predictions depend on the quality, coverage, and representativeness of
  the source data.
- Missing observations and differences in monitoring coverage can affect
  model reliability.
- Geographic validation is important before comparing cities or regions.
- Classification probabilities are model outputs, not calibrated medical
  probabilities.
- The system does not establish causal relationships between pollution
  and health outcomes.
- The application is a technical demonstration, not a certified public
  health decision-support system.

## Project status

The repository contains the data-processing and Machine Learning scripts,
the Lambda inference code, the container definition, and the Streamlit
dashboard.

The AWS inference architecture has been deployed and tested. SageMaker
Studio experimentation has been discontinued to control costs.

Further work can include automated tests, API input validation, systematic
model evaluation, monitoring improvements, and automated deployment.

## Author

**Nada Ben Hammouda**

Biomedical Engineer | AI Engineering | Machine Learning | Cloud Computing

GitHub: [@nadabh98](https://github.com/nadabh98)
