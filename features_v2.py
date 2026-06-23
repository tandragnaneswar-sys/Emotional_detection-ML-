import librosa
import numpy as np

def extract_features(file_path):

    # Load audio
    audio, sr = librosa.load(file_path, duration=3)

    # -------------------------
    # MFCC
    # -------------------------
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    mfcc_mean = np.mean(mfccs.T, axis=0)
    mfcc_std = np.std(mfccs.T, axis=0)

    # -------------------------
    # Chroma
    # -------------------------
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # -------------------------
    # Spectral Contrast
    # -------------------------
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # -------------------------
    # Pitch
    # -------------------------
    pitch = librosa.yin(
        audio,
        fmin=50,
        fmax=300
    )

    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    # -------------------------
    # Energy
    # -------------------------
    rms = librosa.feature.rms(y=audio)[0]

    energy_mean = np.mean(rms)
    energy_std = np.std(rms)

    # -------------------------
    # Zero Crossing Rate
    # -------------------------
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # -------------------------
    # Spectral Centroid
    # -------------------------
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    # -------------------------
    # Spectral Bandwidth
    # -------------------------
    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )
    )

    # -------------------------
    # Final Feature Vector
    # -------------------------
    features = np.hstack([

        # MFCC
        mfcc_mean,      # 20
        mfcc_std,       # 20

        # Chroma
        chroma,         # 12

        # Spectral Contrast
        contrast,       # 7

        # Pitch
        pitch_mean,     # 1
        pitch_std,      # 1

        # Energy
        energy_mean,    # 1
        energy_std,     # 1

        # ZCR
        zcr,            # 1

        # Spectral Features
        centroid,       # 1
        bandwidth       # 1
    ])

    return features