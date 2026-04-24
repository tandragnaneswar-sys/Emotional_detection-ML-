import os
import numpy as np
from features import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset path
dataset_path = "ravdess_data"

X = []
y = []

# Load data
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            features = extract_features(file_path)
            X.append(features)

            # Extract emotion label from filename (RAVDESS format)
            label = file.split("-")[2]
            y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Feature scaling (important for NB stability)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Naive Bayes model
model = GaussianNB()

# Hyperparameter tuning
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)

# Train using GridSearch
grid.fit(X_train, y_train)

# Best model
print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("\nTuned Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))