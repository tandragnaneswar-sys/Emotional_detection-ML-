import joblib
import numpy as np
from features import extract_features


# =========================
# Emotion Mapping
# =========================
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# =========================
# Load saved model
# =========================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")


# =========================
# Prediction Function
# =========================
def predict_audio(file_path):
    # Extract features
    features = extract_features(file_path)

    # Reshape + scale
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    # Get emotion code (like "03")
    emotion_code = encoder.inverse_transform(prediction)[0]

    # Convert to actual emotion name
    emotion_name = emotion_map.get(emotion_code, "Unknown")

    return emotion_name


# =========================
# Test Example
# =========================
file = r"C:\Users\bhanu\OneDrive\Desktop\ML_project\ravdess_data\Actor_18\03-01-05-02-02-01-18.wav"

print("Predicted Emotion:", predict_audio(file))