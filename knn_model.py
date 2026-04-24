import os
import numpy as np
from features import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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



knn = KNeighborsClassifier()



param_grid = {
    "n_neighbors": list(range(3, 21, 2)),
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}


grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)



grid_search.fit(X_train, y_train)


model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)


pred = model.predict(X_test)



print("\nKNN RESULTS\n")

print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))