from flask import Flask, render_template, request, jsonify,redirect, url_for
import joblib
import os
import warnings
import pandas as pd
import glob
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

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

def apply_label_encoders(disease_name, df):
    base_name = disease_name.split(' ')[0].lower()
    encoders = glob.glob(os.path.join(MODELS_DIR, f"{base_name}_*_encoder.pkl"))

    for enc_path in encoders:
        col_name = os.path.basename(enc_path).replace(f"{base_name}_", "").replace("_encoder.pkl", "")
        if col_name in df.columns:
            le = joblib.load(enc_path)
            try:
                df[col_name] = le.transform(df[col_name])
            except ValueError:
                df[col_name] = le.transform([le.classes_[0]] * len(df))
    return df

def prepare_features(df_input, model):
    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        return df_input

    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0.0

    df_input = df_input[model_features].copy()

    for col in df_input.columns:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)

    return df_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_inputs = {k: [v] for k, v in request.form.items()}
        df_input = pd.DataFrame(user_inputs)
        results = {}

        for disease, (model, scaler) in models.items():
            disease_df = prepare_features(df_input.copy(), model)

            disease_df = apply_label_encoders(disease, disease_df)

            try:
                scaler_features = list(scaler.feature_names_in_)
                for col in scaler_features:
                    if col not in disease_df.columns:
                        disease_df[col] = 0.0
                disease_df[scaler_features] = scaler.transform(disease_df[scaler_features])
            except AttributeError:
                pass

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(disease_df)[0][1] * 100
                results[disease] = round(prob, 2)
            else:
                pred = model.predict(disease_df)[0]
                results[disease] = round(float(pred), 2)

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', predictions=[('Error', str(e))])

    top_diseases = sorted(results.items(), key=lambda x: x[1], reverse=True)[:4]

    return render_template('index.html', predictions=top_diseases)

@app.route('/models')
def models_page():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('index.html')
@app.route('/partials/models')
def partial_models():
    return render_template('models.html')

@app.route('/partials/dashboard')
def partial_dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)