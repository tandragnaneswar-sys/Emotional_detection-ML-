# Emotion Detection from Speech using Machine Learning

## Project Overview

This project detects human emotions from speech audio using Machine Learning techniques on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

The system extracts acoustic features from speech signals and classifies them into one of eight emotions:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## Dataset

Dataset Used: RAVDESS

Total Samples: 1440 audio recordings

Emotions:

| Code | Emotion   |
| ---- | --------- |
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

## Feature Engineering

### Initial Feature Set (44 Features)

* MFCC Mean (20)
* Chroma Features (12)
* Spectral Contrast (7)
* Pitch Mean
* Energy Mean
* Zero Crossing Rate
* Spectral Centroid
* Spectral Bandwidth

### Feature Optimization

Several experiments were conducted to improve model performance:

#### Experiment 1: Feature Removal

Removed:

* Spectral Centroid
* Spectral Bandwidth

Result:

* Accuracy decreased from 69.79% to 67.71%

Conclusion:

* Both features contribute useful information and were retained.

#### Experiment 2: MFCC Variability Features

Added:

* MFCC Standard Deviation (20 features)

Result:

* Accuracy improved from 69.79% to 77.78%

Conclusion:

* Temporal variation in speech contains significant emotional information.

#### Experiment 3: Pitch Variability Features

Added:

* Pitch Standard Deviation

Result:

* Accuracy improved further to 78.13%

#### Experiment 4: Energy Variability Features

Added:

* Energy Standard Deviation

Result:

* Accuracy decreased to 77.43%

Conclusion:

* Energy variability did not provide additional discriminative power.

### Final Feature Set (65 Features)

* MFCC Mean (20)
* MFCC Standard Deviation (20)
* Chroma Features (12)
* Spectral Contrast (7)
* Pitch Mean
* Pitch Standard Deviation
* Energy Mean
* Zero Crossing Rate
* Spectral Centroid
* Spectral Bandwidth

---

## Models Evaluated

| Model               | Accuracy |
| ------------------- | -------: |
| SVM                 |   78.13% |
| KNN                 |   63.19% |
| Random Forest       |   59.38% |
| Logistic Regression |   46.53% |
| Decision Tree       |   35.76% |
| Naive Bayes         |   34.38% |
| AdaBoost            |   31.25% |

---

## Emotion Similarity Analysis

Hierarchical Clustering was performed on emotion centroids.

Observations:

* Neutral and Calm formed the closest cluster.
* A separate experiment merged Neutral and Calm into a single class.
* Overall accuracy remained unchanged.

Conclusion:

Although Neutral and Calm are acoustically similar, they were not the primary source of classification error.

---

## Final Results

Best Model: Support Vector Machine (SVM)

Best Accuracy: 78.13%

Best Parameters:

* Kernel: RBF
* C: 10
* Gamma: Scale

---

## My Contributions

* Implemented complete audio feature extraction pipeline using Librosa.
* Performed feature engineering and systematic feature selection experiments.
* Improved baseline accuracy from 69.79% to 78.13%.
* Conducted hierarchical clustering analysis of emotions.
* Evaluated and compared six classical machine learning algorithms.
* Performed hyperparameter tuning using GridSearchCV.
* Analyzed confusion matrices and classification reports to guide model improvements.

---

## Technologies Used

* Python
* Librosa
* NumPy
* Scikit-Learn
* SciPy
* Matplotlib
* Git & GitHub

---

## Future Work

* Delta MFCC Features
* Delta-Delta MFCC Features
* Ensemble Learning Approaches
* Real-Time Emotion Detection
* Deep Learning Models (CNN/LSTM)
