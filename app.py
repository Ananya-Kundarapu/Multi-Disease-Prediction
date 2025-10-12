from flask import Flask, render_template, request
import joblib
import os

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

disease_features = {
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
    try:
        user_inputs = {k: float(v) for k, v in zip(request.form.keys(), request.form.values())}
    except ValueError:
        return render_template('index.html', predictions=[('Error', 'Please enter valid numbers')])

    results = {}

    for disease, (model, scaler) in models.items():
        try:
            features = [user_inputs[f] for f in disease_features[disease]]
        except KeyError:
            return render_template('index.html', predictions=[('Error', f'Missing input for {disease}')])
        
        scaled = scaler.transform([features])
        prob = model.predict_proba(scaled)[0][1]  # Probability of positive class
        results[disease] = round(prob * 100, 2)

    top_diseases = sorted(results.items(), key=lambda x: x[1], reverse=True)[:2]

    return render_template('index.html', predictions=top_diseases)

if __name__ == '__main__':
    app.run(debug=True)