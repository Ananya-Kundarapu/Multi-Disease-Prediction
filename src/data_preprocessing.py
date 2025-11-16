import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    df.replace(["?", "", " ", "nan", "NaN", "NAN"], np.nan, inplace=True)
    df = df.dropna(subset=[target_column])

    numeric_cols = []
    cat_cols = []

    for col in df.columns:
        if col == target_column:
            continue

        try:
            df[col] = pd.to_numeric(df[col])
            numeric_cols.append(col)
        except:
            df[col] = df[col].astype(str).str.lower().str.strip()
            cat_cols.append(col)

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    file_base_name = os.path.basename(file_path).replace('.csv', '')

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        joblib.dump(le, os.path.join(MODELS_DIR, f'{file_base_name}_{col}_encoder.pkl'))

    y = df[target_column]
    if y.dtype == object:
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        joblib.dump(le_target, os.path.join(MODELS_DIR, f'{file_base_name}_target_encoder.pkl'))
    X = df.drop(columns=[target_column])

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{file_base_name}_scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, f'{file_base_name}_features.pkl'))

    return X_scaled, y