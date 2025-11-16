import joblib, os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from data_preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR, DATA_PATH = os.path.join(BASE_DIR, "models"), os.path.join(BASE_DIR, "data", "kidney.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

X, y = preprocess_data(DATA_PATH, "classification")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

min_class_count = min(Counter(y_train).values())
k = 1 if min_class_count <= 2 else 2
smote = SMOTE(random_state=42, k_neighbors=k)
X_train, y_train = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)
nb = GaussianNB()

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("nb", nb)],
    voting="soft"
)
ensemble.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(ensemble, method="sigmoid", cv=3)
calibrated_model.fit(X_train, y_train)

y_pred = calibrated_model.predict(X_test)

print(f"Kidney Disease Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, digits=3))

joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "kidney_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "kidney_features.pkl"))
print("Kidney disease model and feature list saved successfully!")