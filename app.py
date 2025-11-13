from flask import Flask, render_template, request
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

def prepare_features(df_input, disease_name, model, scaler):
    base_name = disease_name.split(' ')[0].lower()
    features_path = os.path.join(MODELS_DIR, f"{base_name}_features.pkl")

    if os.path.exists(features_path):
        expected_features = joblib.load(features_path)
        expected_features = [f for f in expected_features if f != 'id']
    else:
        try:
            expected_features = list(model.feature_names_in_)
        except AttributeError:
            expected_features = list(df_input.columns)

    clean_df = pd.DataFrame(columns=expected_features)
    for col in expected_features:
        if col in df_input.columns:
            clean_df[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0.0)
        elif col.lower() in [c.lower() for c in df_input.columns]:
            matched_col = [c for c in df_input.columns if c.lower() == col.lower()][0]
            clean_df[col] = pd.to_numeric(df_input[matched_col], errors='coerce').fillna(0.0)
        else:
            clean_df[col] = 0.0

    clean_df = clean_df[expected_features]

    try:
        if hasattr(scaler, "feature_names_in_"):
            scaler_features = list(scaler.feature_names_in_)
            for col in scaler_features:
                if col not in clean_df.columns:
                    clean_df[col] = 0.0
            clean_df[scaler_features] = scaler.transform(clean_df[scaler_features])
        else:
            clean_df = scaler.transform(clean_df)
    except Exception as e:
        print(f"⚠️ Scaling issue for {disease_name}: {e}")

    return pd.DataFrame(clean_df, columns=expected_features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_inputs = {}
        for k, v in request.form.items():
            try:
                user_inputs[k] = [float(v) if v.strip() != "" else 0.0]
            except ValueError:
                user_inputs[k] = [0.0]

        df_input = pd.DataFrame(user_inputs)
        results = {}

        for disease, (model, scaler) in models.items():
            df_ready = apply_label_encoders(disease, df_input.copy())
            df_ready = prepare_features(df_ready, disease, model, scaler)
            df_ready = pd.DataFrame(df_ready).fillna(0)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df_ready)[0][1] * 100
            else:
                prob = model.predict(df_ready)[0] * 100

            prob = np.clip(prob, 0, 100)

            if prob < 40:
                risk_label = "Low Risk"
            elif prob < 70:
                risk_label = "Moderate Risk"
            else:
                risk_label = "High Risk"

            results[disease] = (round(prob, 2), risk_label)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return render_template('index.html', predictions=[('Error', str(e))])

    top_diseases = sorted(results.items(), key=lambda x: x[1][0], reverse=True)[:4]
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