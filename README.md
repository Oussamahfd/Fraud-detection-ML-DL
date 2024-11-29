# Fraud Detection Using Machine Learning and Deep Learning

This repository focuses on detecting fraudulent financial transactions using advanced machine learning (ML) and deep learning (DL) techniques. The project involves preprocessing, feature engineering, and the implementation of various models to analyze and predict fraudulent behavior effectively.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Usage Instructions](#usage-instructions)
7. [Technologies Used](#technologies-used)
8. [Future Work](#future-work)
9. [License](#license)

---

## Introduction
Fraudulent transactions significantly affect financial systems. This project applies ML and DL algorithms to detect anomalies in transaction data. The goal is to maximize accuracy and minimize false positives, balancing effectiveness with efficiency.

---

## Dataset
The project uses a dataset generated using PaySim, a financial transaction simulator. The dataset contains:
- **Features**: Transaction type, amount, sender/receiver balance, etc.
- **Target**: Label indicating whether the transaction is fraudulent.

You can find the dataset in the `data/` directory.

---

## Project Structure

Fraud-detection-ML-DL/
├── datasets/                 # Contains datasets
├── Code/            # Jupyter Notebooks for experiments
├── Rapports/              # Python scripts for models and utilities
└── README.md             # Project documentation

---

## Methodology
1. **Data Preprocessing**:
   - Handling missing values, encoding categorical features, and scaling numerical features.
   - Balancing classes using SMOTE (Synthetic Minority Oversampling Technique).

2. **Models Implemented**:
   - **Machine Learning**: K-Nearest Neighbors (KNN), Random Forest (RF), Gradient Boosting.
   - **Deep Learning**: 
     - Autoencoders for anomaly detection.
     - Convolutional Neural Networks (CNNs).
     - Multilayer Perceptrons (MLPs).
     - Long Short-Term Memory (LSTM).

3. **Evaluation Metrics**:
   - Precision, Recall, F1-Score, ROC-AUC, and Matthews Correlation Coefficient (MCC).

---

## Results
Key results from the experiments include:
- **Machine Learning**: Random Forest achieved an F1-score of 0.92.
- **Deep Learning**: Autoencoder outperformed in terms of anomaly detection, with an ROC-AUC of 0.95.

Visualization plots (e.g., ROC curves, confusion matrices) are available in the `results/` directory.

---


## Technologies Used
- **Languages**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Git

---

## Future Work
- Implementing more sophisticated models such as transformers.
- Optimizing hyperparameters using advanced search methods.
- Deploying the best-performing model as a web service.


