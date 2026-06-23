import os
import numpy as np
import pandas as pd

from features import extract_features

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


dataset_path = "Ravadess"

emotion_names = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

emotion_features = {key: [] for key in emotion_names.keys()}

# Load data
for root, dirs, files in os.walk(dataset_path):
    for file in files:

        if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            emotion = file.split("-")[2]

            features = extract_features(file_path)

            emotion_features[emotion].append(features)

# Compute centroid for each emotion
centroids = []
labels = []

for emotion_code in emotion_names:

    centroid = np.mean(emotion_features[emotion_code], axis=0)

    centroids.append(centroid)

    labels.append(emotion_names[emotion_code])

centroids = np.array(centroids)

# Hierarchical clustering
Z = linkage(
    centroids,
    method="ward"
)

# Plot dendrogram
plt.figure(figsize=(10, 6))

dendrogram(
    Z,
    labels=labels
)

plt.title("Hierarchical Clustering of Emotions")
plt.xlabel("Emotion")
plt.ylabel("Distance")

plt.tight_layout()
plt.savefig("emotion_dendrogram.png")
print("Dendrogram saved as emotion_dendrogram.png")
print("Loading dataset...")