from flask import Flask, render_template, request
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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

# Feature order must match training
feature_order = {
    'Diabetes': ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
    'Heart Disease': ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],
    'Kidney Disease': ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv'],
    'Liver Disease': ['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A_G']
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Convert form inputs to floats
    try:
        user_inputs = {k: float(v) for k, v in zip(request.form.keys(), request.form.values())}
    except ValueError:
        return render_template('index.html', predictions=[('Error', 'Please enter valid numbers')])

    results = {}

    # Loop over all diseases
    for disease, (model, scaler) in models.items():
        features = []
        for f in feature_order[disease]:
            features.append(user_inputs.get(f, 0))  # Use 0 if missing

        try:
            scaled = scaler.transform([features])
            prob = model.predict_proba(scaled)[0][1]  # Probability of positive class
            results[disease] = round(prob * 100, 2)
        except Exception as e:
            results[disease] = f"Error: {str(e)}"

    top_diseases = sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)[:4]

    return render_template('index.html', predictions=top_diseases)

if __name__ == '__main__':
    app.run(debug=True)