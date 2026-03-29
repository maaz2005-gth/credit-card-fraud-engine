"""
train_lodo.py
=============
Leave-One-Domain-Out (LODO) validation for the Generalized Fraud Engine.
Trains on one dataset (domain), evaluates on the completely unseen domain.
Logs Accuracy, Precision, Recall, F1-Score, and AUC.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from features import engineer_features
from sampling import apply_oversampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob):
    """Calculates all requested performance metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1_Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob)
    }

def run_lodo_validation():
    logger.info("=" * 60)
    logger.info("STARTING LODO (Leave-One-Domain-Out) CROSS-VALIDATION")
    logger.info("Metrics tracked: Accuracy, Precision, Recall, F1, AUC")
    logger.info("=" * 60)

    # 1. Load data
    logger.info("Loading harmonized data...")
    df = pd.read_csv("data/harmonized.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    domains = df["source_domain"].unique()
    logger.info("Found domains: %s", domains)
    
    lodo_results = {}

    # 2. LODO Loop
    for test_domain in domains:
        logger.info("\n" + "=" * 50)
        logger.info(f"FOLD: Testing on unseen domain -> [{test_domain}]")
        logger.info("=" * 50)
        
        train_domains = [d for d in domains if d != test_domain]
        logger.info(f"  Training on domains -> {train_domains}")
        
        # Split Data
        df_train = df[df["source_domain"].isin(train_domains)].copy()
        df_test = df[df["source_domain"] == test_domain].copy()
        
        # Apply Feature Engineering
        logger.info("  Engineering features...")
        X_train, y_train, _, train_stats = engineer_features(df_train, fit=True)
        X_test, y_test, _, _ = engineer_features(df_test, fit=False, stats=train_stats)
        
        # Apply WGAN-GP Oversampling
        X_train_res, y_train_res = apply_oversampling(X_train, y_train, target_ratio=0.15)
        
        # Train XGBoost
        logger.info("  Training XGBoost Classifier...")
        scale_pos = (y_train_res == 0).sum() / max(1, (y_train_res == 1).sum())
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)
        
        # Predict & Evaluate
        logger.info("  Evaluating on completely unseen domain [%s]...", test_domain)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        lodo_results[test_domain] = metrics
        
        logger.info("-" * 40)
        for metric, val in metrics.items():
            logger.info(f"  {metric:10s} : {val:.4f}")
        logger.info("-" * 40)

    # 3. Validation Summary
    print("\n\n" + "=" * 60)
    print("FINAL LODO CROSS-DOMAIN METRICS SUMMARY")
    print("=" * 60)
    
    summary_df = pd.DataFrame(lodo_results).T
    print(summary_df.round(4).to_string())
    
    # Save the model trained on all data (optional, for the final API)
    # logger.info("\nSaving final model...")

if __name__ == "__main__":
    run_lodo_validation()
