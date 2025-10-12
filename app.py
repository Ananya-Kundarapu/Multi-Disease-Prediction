from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models and scalers
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
    return render_template('index.html', form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    # Store user inputs and keep for re-rendering
    try:
        form_values = {k: float(v) for k, v in request.form.items()}
    except ValueError:
        return render_template('index.html', predictions=[('Error', 'Please enter valid numbers')],
                               form_values=request.form)

    results = {}

    # Map inputs to model features for each disease
    disease_feature_map = {
        'Diabetes': ['glucose', 'insulin', 'bmi', 'age'],
        'Heart Disease': ['bp', 'cholesterol', 'max_heart_rate', 'oldpeak'],
        'Kidney Disease': ['sg', 'al', 'su', 'bgr'],
        'Liver Disease': ['sgpt', 'sgot', 'alkphos', 'bilirubin']
    }

    for disease, (model, scaler) in models.items():
        try:
            features = [form_values[f] for f in disease_feature_map[disease]]
        except KeyError:
            return render_template('index.html', predictions=[('Error', f'Missing input for {disease}')],
                                   form_values=form_values)

        scaled = scaler.transform([features])
        prob = model.predict_proba(scaled)[0][1]
        results[disease] = round(prob * 100, 2)

    top_diseases = sorted(results.items(), key=lambda x: x[1], reverse=True)[:2]

    return render_template('index.html', predictions=top_diseases, form_values=form_values)

if __name__ == '__main__':
    app.run(debug=True)
