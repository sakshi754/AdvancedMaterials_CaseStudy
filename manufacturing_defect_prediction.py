# manufacturing_defect_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

# Generate synthetic dataset
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    temperature = np.random.normal(loc=75, scale=5, size=n_samples)
    pressure = np.random.normal(loc=30, scale=2, size=n_samples)
    humidity = np.random.normal(loc=50, scale=10, size=n_samples)
    vibration = np.random.normal(loc=5, scale=1, size=n_samples)
    speed = np.random.normal(loc=100, scale=10, size=n_samples)
    defect_label = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

    data = pd.DataFrame({
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'vibration': vibration,
        'speed': speed,
        'defect_label': defect_label
    })
    return data

# Load and split the data
data = generate_synthetic_data()
X = data.drop('defect_label', axis=1)
y = data['defect_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Anomaly Detection (Unsupervised Insight)
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = anomaly_detector.fit_predict(X)
data['anomaly'] = (anomaly_labels == -1).astype(int)

# Supervised Model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# MLflow Logging
mlflow.set_experiment("Manufacturing Defect Prediction")
with mlflow.start_run():
    mlflow.log_params({"model": "GradientBoostingClassifier", "test_size": 0.2})
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")
