import json

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

from _aux import features as F

DEFAULT_RISK_THRESHOLD = 0.8

app = Flask(__name__)

pipeline = joblib.load("_aux/pipeline.joblib.gz")


@app.route("/")
def index():
    return jsonify(
        {
            "notice": "Prediction requests should be directed to route '/api/v1/default_risk/predict'"
        }
    )


@app.route("/health/ready", methods=["GET"])
def get_health_ready():
    return Response(status=200)


@app.route("/api/v1/default_risk/predict", methods=["POST"])
def predict():

    try:
        header = request.headers["Authorization"]
        if header != "klarna-case-study":
            raise KeyError
    except KeyError:
        return jsonify({"type": "unauthorized_access"}), 401

    data = request.get_json()
    if data is None:
        return (
            jsonify(
                {"type": "json_missing_from_request", "received": json.dumps(data)}
            ),
            400,
        )

    try:
        df = pd.read_json(data, orient="records")
    except ValueError:
        return (
            jsonify(
                {"type": "data_incorrect_format", "expected": F.expected_payload()}
            ),
            400,
        )

    try:
        df = df[F.required_columns()]
    except KeyError:
        missing = [col for col in F.required_columns() if col not in df.columns]
        return (
            jsonify(
                {
                    "type": "required_columns_missing_from_data",
                    "missing_columns": json.dumps(missing),
                }
            ),
            400,
        )

    prediction = df[["uuid"]].assign(
        pd=pipeline.predict_proba(df)[:, 1],
        flag_default=lambda df: (df["pd"] > DEFAULT_RISK_THRESHOLD).astype(int),
    )

    payload = prediction.to_json(orient="records")

    return payload, 200


if __name__ == "__main__":
    app.run()
