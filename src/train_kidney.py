# src/train_kidney.py
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data

# Preprocess data
X, y = preprocess_data('data/kidney.csv', 'classification')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)
print("Kidney Disease Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(ensemble, 'models/kidney_model.pkl')