# app/api.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
from model import load_model

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    prediction = model.predict(np.array(data).reshape(1, -1)).tolist()
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
