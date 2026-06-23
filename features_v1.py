import librosa
import numpy as np

def extract_features(file_path):

    # Load audio
    audio, sr = librosa.load(file_path, duration=3)

    # MFCC
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)

    # Chroma
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Spectral Contrast
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Pitch
    pitch = librosa.yin(
        audio,
        fmin=50,
        fmax=300
    )

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    # Energy
    energy = np.mean(
        librosa.feature.rms(y=audio)
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # Spectral Centroid
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    # Spectral Bandwidth
    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    # Combine Features
    features = np.hstack([

        # Original Features
        mfcc_mean,      # 20
        chroma,         # 12
        contrast,       # 7
        pitch_mean,     # 1
        energy,         # 1
        zcr,            # 1
        centroid,       # 1
        bandwidth,      # 1

        # Added Features
        mfcc_std,       # 20
        pitch_std       # 1

    ])

    return features