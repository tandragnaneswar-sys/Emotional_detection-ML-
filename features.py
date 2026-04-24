import librosa
import numpy as np

def extract_features(file_path):

    # Load audio
    audio, sr = librosa.load(file_path, duration=3)

    # MFCC (20)
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20).T,
        axis=0
    )

    # Chroma (12)
    chroma = np.mean(
        librosa.feature.chroma_stft(y=audio, sr=sr).T,
        axis=0
    )

    # Spectral Contrast (7)
    contrast = np.mean(
        librosa.feature.spectral_contrast(y=audio, sr=sr).T,
        axis=0
    )

    # Pitch
    pitch = librosa.yin(audio, fmin=50, fmax=300)
    pitch_mean = np.mean(pitch)

    # Energy
    energy = np.mean(librosa.feature.rms(y=audio))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    # Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    # Spectral Bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    # Combine all features
    features = np.hstack([
        mfcc,
        chroma,
        contrast,
        pitch_mean,
        energy,
        zcr,
        centroid,
        bandwidth
    ])

    return features