import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(file_path, target_column):
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Fill numeric NaNs with median
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, file_path.replace('.csv', f'_{col}_encoder.pkl'))

    # Encode target if it is categorical
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        joblib.dump(le_target, file_path.replace('.csv', '_labelencoder.pkl'))

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
    
    scaler_path = os.path.join('models', os.path.basename(file_path).replace('.csv', '_scaler.pkl'))
    joblib.dump(scaler, scaler_path)

    return X_scaled, y
