import os
import numpy as np
from features import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


dataset_path = "ravdess_data"

X = []
y = []


# Load dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            # Extract emotion label
            emotion = file.split("-")[2]

            # Extract features
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


#  IMPORTANT: SVM needs scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# SVM Model
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


# Train model
grid_search.fit(X_train, y_train)


# Best model
model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)


# Predictions
pred = model.predict(X_test)


print("\nSVM RESULTS\n")

print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))


# 🎯 Predict emotion for new audio

new_audio = "C:/Users/bhanu/OneDrive/Desktop/ML_project/ravdess_data/Actor_12/03-01-01-01-01-01-12.wav"

# Extract features
new_features = extract_features(new_audio)

# Convert and scale
new_features = np.array(new_features).reshape(1, -1)
new_features = scaler.transform(new_features)

# Predict
prediction = model.predict(new_features)

# Decode label
emotion = encoder.inverse_transform(prediction)

print("\nPredicted Emotion:", emotion[0])