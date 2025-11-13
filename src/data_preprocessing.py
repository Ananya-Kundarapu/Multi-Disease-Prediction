import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    file_base_name = os.path.basename(file_path).replace('.csv', '')

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(MODELS_DIR, f'{file_base_name}_{col}_encoder.pkl'))

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        joblib.dump(le_target, os.path.join(MODELS_DIR, f'{file_base_name}_target_encoder.pkl'))

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{file_base_name}_scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, f'{file_base_name}_features.pkl'))

    return X_scaled, y