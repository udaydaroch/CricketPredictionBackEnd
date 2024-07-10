from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5174"}})

model = joblib.load('random_forest_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

matches_file = 'combined_matches.jsonl'
matches_df = pd.read_json(matches_file, lines=True)

@app.route('/matches', methods=['GET'])
def get_match_options():
    teams = sorted(set(matches_df['team1']).union(set(matches_df['team2'])))
    venues = sorted(matches_df['venue'].unique())

    return jsonify({
        'teams': teams,
        'venues': venues
    })

@app.route('/', methods=["GET"])
def helloWorld():
    return jsonify("Hello World")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    input_df = pd.DataFrame([data])

    input_encoded = pd.get_dummies(input_df).reindex(columns=model.feature_names_in_, fill_value=0)

    probabilities = model.predict_proba(input_encoded)
    predicted_winner = label_encoder.inverse_transform([model.predict(input_encoded)[0]])
    predicted_probability = probabilities[0][model.classes_ == label_encoder.transform(predicted_winner)[0]][0]

    response = {
        'predicted_winner': predicted_winner[0],
        'predicted_probability': predicted_probability * 100
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
