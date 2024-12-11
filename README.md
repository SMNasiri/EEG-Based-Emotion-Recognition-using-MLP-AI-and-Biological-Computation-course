# EEG-Based-Emotion-Recognition-using-MLP
EEG-Based Emotion Recognition project that employs Multi-Layer Perceptron (MLP) for classifying emotions based on EEG signals. The project includes feature extraction, feature selection, and training a neural network to classify emotions as positive or negative.

This project focuses on recognizing emotions using EEG signals and a Multi-Layer Perceptron (MLP) neural network. The goal is to classify EEG data into two emotion classes: **positive** and **negative**. The project involves preprocessing EEG data, extracting meaningful features, selecting the most relevant features, and training a neural network for classification.

---

## **Project Overview**
- **Objective**: Classify EEG signals into two emotion classes using MLP.
- **Data Source**: EEG signals recorded from 59 channels during experiments designed to elicit specific emotions.
- **Feature Extraction**:
  - Statistical features (mean, variance, etc.).
  - Frequency domain features (Fourier Transform-based).
- **Feature Selection**:
  - Fisher's Criterion for identifying the most discriminative features.
  - k-fold Cross-Validation for evaluating feature subsets.
- **Classification**:
  - MLP neural network designed and trained to classify emotions.
  - Hyperparameters such as activation functions, number of layers, and neurons per layer are tuned for optimal performance.

---

## **Workflow**
1. **Data Collection**:
   - EEG signals recorded from 59 channels, with a sampling rate of 1000 Hz.
   - Data divided into training and test sets (550 samples for training, 159 for testing).
2. **Feature Engineering**:
   - Extract statistical and frequency-based features from EEG signals.
   - Normalize features and select the most effective subset using Fisher's Criterion.
3. **Model Training**:
   - Train an MLP neural network using the selected features.
   - Use k-fold cross-validation to fine-tune the model and evaluate its performance.
4. **Evaluation**:
   - Measure classification accuracy on the test set.
   - Compare results with other classifiers (e.g., RBF networks).

---

## **Key Features**
- **Feature Extraction**: Comprehensive set of statistical and frequency-based features for effective EEG signal representation.
- **Feature Selection**: Use of Fisher's Criterion and k-fold cross-validation to select the most relevant features.
- **MLP Implementation**: Custom-designed neural network optimized for EEG-based emotion recognition.

---

## **Setup and Usage**
### **Requirements**
- MATLAB R2020b or later.
