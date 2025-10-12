from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Ensure models and scalers paths are correct
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load all models and their scalers
models = {
    'Diabetes': (
        joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl')),
        joblib.load(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'))
    ),
    'Heart Disease': (
        joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl')),
        joblib.load(os.path.join(MODELS_DIR, 'heart_scaler.pkl'))
    ),
    'Kidney Disease': (
        joblib.load(os.path.join(MODELS_DIR, 'kidney_model.pkl')),
        joblib.load(os.path.join(MODELS_DIR, 'kidney_scaler.pkl'))
    ),
    'Liver Disease': (
        joblib.load(os.path.join(MODELS_DIR, 'liver_model.pkl')),
        joblib.load(os.path.join(MODELS_DIR, 'liver_scaler.pkl'))
    )
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Convert all form values to float
    try:
        features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('index.html', predictions=[('Error', 'Please enter valid numbers')])

    results = {}

    # Predict probability for each disease
    for disease, (model, scaler) in models.items():
        scaled = scaler.transform([features])
        prob = model.predict_proba(scaled)[0][1]  # Probability of positive class
        results[disease] = round(prob * 100, 2)

    # Sort and take top 2 predictions
    top_diseases = sorted(results.items(), key=lambda x: x[1], reverse=True)[:2]

    return render_template('index.html', predictions=top_diseases)

if __name__ == '__main__':
    app.run(debug=True)
