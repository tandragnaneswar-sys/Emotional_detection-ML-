import os
import numpy as np
from features import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



dataset_path = "ravdess_data"

X = []
y = []


for root, dirs, files in os.walk(dataset_path):

    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion = file.split("-")[2]

            features = extract_features(file_path)

            X.append(features)
            y.append(emotion)



X = np.array(X)

print("Total samples:", len(X))


encoder = LabelEncoder()
y = encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



log_reg = LogisticRegression(max_iter=5000)



param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs"],
    "penalty": ["l2"]
}



grid_search = GridSearchCV(
    log_reg,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)



grid_search.fit(X_train, y_train)


model = grid_search.best_estimator_

print("\nBest Parameters:", grid_search.best_params_)


print("Iterations used by Logistic Regression:", model.n_iter_)


pred = model.predict(X_test)


print("\nLOGISTIC REGRESSION RESULTS\n")

print("Accuracy:", accuracy_score(y_test, pred))


print("\nClassification Report:\n")
print(classification_report(y_test, pred))


print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))