import json
import os
import boto3
import joblib
import numpy as np

S3_BUCKET = "ai-health-aws-nadabh-2026"
MODEL_KEY = "models/xgboost_health_risk_final.joblib"
MODEL_PATH = "/tmp/model.joblib"

s3 = boto3.client("s3")
_model = None

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

CLASSES = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            s3.download_file(S3_BUCKET, MODEL_KEY, MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model

def lambda_handler(event, context):
    try:
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        missing = [f for f in FEATURES if f not in body]
        if missing:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "error": "Missing required features",
                    "missing": missing
                })
            }

        values = [float(body[f]) for f in FEATURES]
        model = load_model()
        prediction = int(model.predict([values])[0])

        result = {"risk_level": CLASSES.get(prediction, "UNKNOWN")}

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([values])[0]
            result["probabilities"] = {
                CLASSES[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }

    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)})
        }
    except Exception:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Prediction failed"})
        }
