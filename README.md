# 🧠 Multi-Disease Prediction using Ensemble Learning

This project predicts the likelihood of multiple diseases — **Diabetes, Heart Disease, Kidney Disease, and Liver Disease** — using **machine learning ensemble models** such as **Random Forest, SVM, XGBoost,** and **KNN**.
It combines the predictions of all models to identify the most probable disease for a given user’s medical inputs.

---

## 🚀 Tech Stack

* **Python (Flask)** – Web Framework
* **scikit-learn, XGBoost** – Machine Learning
* **HTML, CSS** – Frontend UI
* **Joblib** – Model persistence and loading

---

## 🧩 Features

* Predicts multiple diseases using trained ML models
* Ensemble-based approach for higher accuracy
* Simple Flask web interface for predictions
* Takes common health inputs (e.g., Glucose, Blood Pressure, Age, BMI, Cholesterol, etc.)
* Displays **the most probable disease** or **Healthy** status

---

## 📁 Project Structure

```
Multi-Disease-Prediction/
│
├── data/               # Datasets for all diseases
├── models/             # Saved models & encoders
├── src/                # Training scripts
├── static/             # CSS files
├── templates/          # HTML templates
├── app.py              # Flask main app
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🏗️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ananya-Kundarapu/Multi-Disease-Prediction.git
cd Multi-Disease-Prediction
```

### 2️⃣ Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
python app.py
```

Then open your browser at 👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## ⚙️ How It Works

1. User enters basic medical details (like Glucose, Cholesterol, etc.)
2. Inputs are preprocessed and scaled
3. Each model (Diabetes, Heart, Kidney, Liver) predicts separately
4. Ensemble logic identifies which disease has the **highest probability**
5. The final result is displayed on the screen

---

## 📊 Model Performance (Approx.)

| Disease        | Accuracy |
| -------------- | -------- |
| Diabetes       | ~80%     |
| Heart Disease  | ~83%     |
| Kidney Disease | ~85%     |
| Liver Disease  | ~82%     |

---

## 🧠 Optional: Re-train Models

To retrain models using new data:

```bash
cd src
python train_diabetes.py
python train_heart.py
python train_kidney.py
python train_liver.py
```