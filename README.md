# Credit Card Fraud Detection Engine

An end-to-end machine learning system for real-time credit card fraud detection. The project combines fraud-focused feature engineering, synthetic minority data generation using WGAN-GP, XGBoost classification, and a FastAPI-based inference service deployed to the cloud.

---

## Project Overview

Credit card fraud detection presents two major challenges:

1. Severe class imbalance, where fraudulent transactions represent only a small fraction of all transactions.
2. Real-time decision requirements, where predictions must be generated with minimal latency.

This project addresses both challenges by combining behavioral feature engineering, synthetic fraud generation, and a production-style inference pipeline.

---

## Key Features

* End-to-end fraud detection workflow
* Advanced feature engineering for transaction behavior analysis
* WGAN-GP-based fraud sample generation
* XGBoost classification model
* FastAPI inference API
* Real-time velocity tracking
* Interactive frontend dashboard
* Cloud deployment on Render

---

## Repository Structure

### /src

Core machine learning pipeline and training workflows.

#### data_harmonizer.py

Standardizes raw transaction datasets into a unified schema for downstream processing.

#### features.py

Generates model features including:

* Transaction amount normalization
* Cyclical time encoding
* 24-hour transaction velocity
* 24-hour spending behavior
* Time since previous transaction

#### sampling.py

Implements fraud class augmentation techniques, including:

* Adversarial Autoencoders (AAE)
* Wasserstein GAN with Gradient Penalty (WGAN-GP)

#### train_single_domain.py

Training pipeline responsible for:

* Data preparation
* Synthetic fraud generation
* Model training
* Hyperparameter tuning
* Model serialization

---

### /API

Backend inference service.

#### main.py

Responsible for:

* Loading trained model artifacts
* Real-time feature computation
* Velocity metric tracking
* Prediction serving
* Frontend hosting

---

### /frontend

User interface for interacting with the fraud detection system.

#### index.html

Application layout.

#### style.css

Dashboard styling.

#### app.js

Frontend logic and API communication.

---

## Machine Learning Pipeline

### Feature Engineering

Behavioral features are generated to capture transaction patterns that are commonly associated with fraudulent activity:

* Transaction frequency
* Spending velocity
* Time-based activity patterns
* Historical user behavior

### Class Imbalance Handling

Fraudulent transactions represent a small minority of the dataset.

To improve minority class representation, WGAN-GP is used to generate synthetic fraud samples, creating a more balanced training set for model learning.

### Model Selection

XGBoost was selected due to:

* Strong performance on tabular data
* Robust handling of non-linear relationships
* Fast inference speed
* High interpretability through feature importance analysis

---

## Model Performance

Evaluation on a hold-out test set produced the following results:

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 99.82% |
| AUC-ROC   | 99.58% |
| Precision | 92.06% |
| Recall    | 75.27% |
| F1 Score  | 82.82% |

### Inference Performance

Average prediction latency:

~45 ms per transaction

---

## Deployment

The application is deployed using Render.

Deployment includes:

* FastAPI backend
* XGBoost model serving
* Static dashboard hosting
* Automated dependency installation through requirements.txt

Server startup:

uvicorn API.main:app --host 0.0.0.0 --port 10000

---

## Future Improvements

* Redis-based distributed velocity tracking
* Real-time streaming with Kafka
* Explainable AI using SHAP
* Model monitoring and drift detection
* Automated retraining pipelines
* Docker containerization
* CI/CD integration

---

## Tech Stack

Machine Learning:

* Python
* XGBoost
* Scikit-Learn
* Pandas
* NumPy

Synthetic Data Generation:

* PyTorch
* WGAN-GP
* Adversarial Autoencoders

Backend:

* FastAPI
* Uvicorn

Frontend:

* HTML
* CSS
* JavaScript

Deployment:

* Render

---

## Disclaimer

This project is intended for educational and portfolio purposes. Production-grade fraud detection systems typically incorporate additional data sources, monitoring infrastructure, regulatory controls, and large-scale distributed processing capabilities.
