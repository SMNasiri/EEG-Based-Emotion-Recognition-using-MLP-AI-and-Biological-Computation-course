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

