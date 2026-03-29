# Credit Card Fraud Detection Engine

**Real-time Full-Stack Fraud Detection Engine**

## 1. Executive Summary
This repository contains an end-to-end, full-stack Enterprise Machine Learning architecture. We transformed a highly imbalanced, physical Point-of-Sale (POS) dataset into a fully functional, cloud-deployed XGBoost Fraud Engine. 

The system leverages Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP) to synthesize fraudulent behavior, an ultra-low latency FastAPI Python inference server that maintains rolling user velocity metrics in RAM, and a custom Vanilla JS Dashboard for real-time human interaction.

## 2. Project Directory Architecture

### Root Repository
*   **.gitignore**: Engineered to block gigabyte-scale raw dataset CSVs (`data/*.csv`) and MLflow tracking registries (`mlruns/`, `mlflow.db`). Note: Production model weights (`API/models/*.pkl`) are explicitly natively tracked for cloud deployment via override flags.
*   **requirements.txt**: The baseline dependency blueprint for the Render Cloud Server, specifying required C-level packages (`fastapi`, `xgboost`, `scikit-learn`, `pandas`).

### /src - The Machine Learning Core
This module handles offline extraction, transformation, tuning, and generation.
*   **data_harmonizer.py**: Maps desperate schema structures into a unified pandas pipeline, optimizing purely for physical POS topologies.
*   **features.py**: Handles Cyclical Time Encoding and scaling via Scikit-Learn's `QuantileTransformer` to Gaussianize skewed transaction amounts. Generates behavioral rules: `velocity_24h`, `spend_24h`, and `time_since_last_txn`. 
*   **sampling.py**: Handles the generational minority class pipeline. Implements Adversarial Autoencoders (AAE) and Wasserstein GANs (WGAN-GP). WGAN-GP is used to synthesize highly realistic mathematical clones of fraudulent transactions without mode-collapsing.
*   **run_ga.py & train_lodo.py**: Implements dynamic hyperparameter evolution via Genetic Algorithms and Leave-One-Domain-Out cross-validation. 
*   **train_single_domain.py**: The definitive training loop. Parses harmonized output, subjects it to the WGAN-GP oversampler, feeds the balanced arrays into the XGBoost Classifier, and serializes the optimal model weights directly to `API/models/`.

### /API - The Web Server Engine
*   **API/main.py**: An Enterprise-grade FastAPI instance handling Model Booting (loading `fraud_model.pkl` to RAM), In-Memory Mock Redis (tracking active user velocity spikes on the fly), and Frontend Web Hosting (mounting the static dashboard).

### /frontend - The Dynamic Dashboard
*   **index.html & style.css**: A responsive, dark-mode dashboard built with standard DOM logic. 
*   **app.js**: Establishes the asynchronous connection to the backend FastAPI via relative routing (`/predict`).

## 3. Performance Metrics & Mitigation Strategies

After training WGAN-GP to oversample the 0.5% fraud minority class and tuning XGBoost, we extracted the definitive hold-out test set metrics:

| Metric | Score | Translation |
| :--- | :--- | :--- |
| Accuracy | 99.82% | Baseline prediction accuracy. |
| AUC-ROC | 99.58% | Clean classification boundary separation. |
| Precision | 92.06% | Exactitude of positive fraud flags (minimizing false positives). |
| Recall | 75.27% | Total capture rate of anomalous events. |
| F1 Score | 82.82% | Harmonic mean proving robust WGAN-GP localization. |

**Inference Performance:** ~45.00 ms. 

## 4. Cloud Deployment Architecture

The codebase is securely deployed using Render Web Services.

**Containerization & Dependencies:**
Render spins up a dedicated Linux container parsing `requirements.txt`. The deployment loads ~400MB of data science libraries natively into memory, bypassing the severe 250MB size restrictions typical in Serverless architectures. 

**Startup Configuration:**
The ASGI server boots via standard Uvicorn execution bound to port 10000:
`uvicorn API.main:app --host 0.0.0.0 --port 10000`
