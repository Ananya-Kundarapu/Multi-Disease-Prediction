import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    file_base_name = os.path.basename(file_path).replace('.csv', '')
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoder_path = os.path.join('models', f'{file_base_name}_{col}_encoder.pkl')
        joblib.dump(le, encoder_path)

    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        target_encoder_path = os.path.join('models', f'{file_base_name}_labelencoder.pkl')
        joblib.dump(le_target, target_encoder_path)

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
    
    scaler_path = os.path.join('models', f'{file_base_name}_scaler.pkl')
    joblib.dump(scaler, scaler_path)

    return X_scaled, y