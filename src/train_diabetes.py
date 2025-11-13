import joblib, os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from data_preprocessing import preprocess_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR, DATA_PATH = os.path.join(BASE_DIR, "models"), os.path.join(BASE_DIR, "data", "diabetes.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

X, y = preprocess_data(DATA_PATH, "Outcome")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
lr = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lr", lr), ("knn", knn)], voting="soft")
ensemble.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(ensemble, method="sigmoid", cv=3)
calibrated_model.fit(X_train, y_train)

y_pred = calibrated_model.predict(X_test)
print(f"Diabetes Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, digits=3))

joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "diabetes_features.pkl"))
print("Diabetes ensemble model and feature list saved successfully!")