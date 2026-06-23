import os
import numpy as np
from features_v1 import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

dataset_path = "Ravadess"

X = []
y = []

# Load dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion = file.split("-")[2]

            features = extract_features(file_path)

            X.append(features)
            y.append(emotion)

# Convert to numpy
X = np.array(X)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM
svm = SVC()

# Hyperparameter tuning
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    svm,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)

# Prediction
pred = model.predict(X_test)

print("\nSVM RESULTS\n")

print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

# Save model
joblib.dump(model, "svm_model_v1_1.pkl")
joblib.dump(scaler, "scaler_v1_1.pkl")
joblib.dump(encoder, "encoder_v1_1.pkl")

print("\nModel Saved Successfully!")