import joblib
import json
from flask import Flask, jsonify, request, Response
import pandas as pd
from _aux import features as F

DEFAULT_RISK_THRESHOLD = .8

app = Flask(__name__)
app.config["DEBUG"] = True

pipeline = joblib.load("ml_artifacts/preprocessor.joblib.gz")


@app.route('/')
def index():
    return jsonify(
        {"notice": "Prediction requests should be directed to route '/api/v1/default_risk/predict'"}
    )


@app.route("/health/ready", methods=["GET"])
def get_health_ready():
    return Response(status=200)


@app.route('/api/v1/default_risk/predict_one', methods=["POST"])
def predict_one():
    try:
        request_json = request.get_json()
    except AttributeError:
        return jsonify({"type": "json_missing_from_request"}), 400
    
    try:
        data = request_json["data"]
    except KeyError:
        return jsonify({"type": "data_missing_from_json"}), 400

    df = pd.read_json(data, orient="records")

    prediction = pipeline.predict(df)
    
    return jsonify(
        {
            "uuid": df["uuid"],
            "pd": prediction,
            "is_default": prediction > DEFAULT_RISK_THRESHOLD
        }
    ), 200


@app.route('/api/v1/default_risk/predict_many', methods=["POST"])
def predict_many():
    try:
        request_json = request.get_json()
    except AttributeError:
        return jsonify({"type": "json_missing_from_request"}), 400
    
    try:
        data = request_json["data"]
    except KeyError:
        return jsonify({"type": "data_missing_from_json"}), 400

    df = pd.read_json(data, orient="records")

    prediction = pipeline.predict(df)
    
    return jsonify(
        {
            "uuid": df["uuid"],
            "pd": prediction,
            "is_default": prediction > DEFAULT_RISK_THRESHOLD
        }
    ), 200


if __name__ == "__main__":
    app.run()