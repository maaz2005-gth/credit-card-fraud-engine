"""
train_single_domain.py
======================
Final Showdown: Runs the primary Sparkov dataset through an optimized pipeline,
evaluating XGBoost trained on WGAN-GP vs Autoencoder (AAE) oversampling.
Saves the winning model for the API.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from features import engineer_features
from sampling import apply_oversampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, name):
    logger.info(f"Evaluating {name} Model with Optimal Threshold...")
    y_prob = model.predict_proba(X_test)[:, 1]

    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (y_prob > t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1, best_thresh = score, t
            
    logger.info(f"  Best Threshold: {best_thresh:.2f}")
    y_pred = (y_prob > best_thresh).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob)
    }
    return metrics, y_pred

def main():
    logger.info("=" * 70)
    logger.info("SINGLE-DOMAIN HIGHEST PERFORMANCE FINAL SHOWDOWN (GAN vs AE)")
    logger.info("=" * 70)

    # 1. Load Data
    logger.info("Loading Sparkov domain data...")
    df = pd.read_csv("data/harmonized.csv", low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["source_domain"] == "sparkov_us_pos"].copy()

    # 2. Train/Test Split
    logger.info("Splitting 80/20 train/test...")
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["is_fraud"], random_state=42)

    # 3. Universal Behavioral Feature Engineering & Scaling
    logger.info("Engineering & Normalizing Features (QuantileTransformer)...")
    X_train, y_train, _, train_stats = engineer_features(df_train, fit=True)
    X_test,  y_test,  _, _           = engineer_features(df_test, fit=False, stats=train_stats)

    # We will test two methods
    methods = ["GAN", "AE"]
    results = {}
    models = {}

    for method in methods:
        logger.info("\n" + "-" * 50)
        logger.info(f"PIPELINE: {method}")
        logger.info("-" * 50)
        
        # 4. Oversampling
        X_train_res, y_train_res = apply_oversampling(X_train, y_train, method=method, target_ratio=0.15)

        # 5. Train XGBoost
        logger.info(f"Training XGBoost on {method} data...")
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)
        models[method] = model

        # 6. Evaluate
        metrics, _ = evaluate_model(model, X_test, y_test, method)
        results[method] = metrics
        
        for k, v in metrics.items():
            logger.info(f"  {k:12s} : {v:.4f}")

    # 7. Compare and Select Winner
    print("\n\n" + "=" * 60)
    print("FINAL PERFORMANCE METRICS COMPARISON")
    print("=" * 60)
    res_df = pd.DataFrame(results).T
    print(res_df.round(4).to_string())

    # We use F1 score to determine the absolute best model
    winner = res_df["F1 Score"].idxmax()
    
    # 7b. Feature Selection / Explainability on Winner
    print("\n" + "=" * 60)
    print(f"XGBOOST FEATURE IMPORTANCES ({winner} MODEL)")
    print("=" * 60)
    best_model = models[winner]
    importances = best_model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
    print(feat_imp.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"WINNER DECLARED: {winner} Model (Highest F1 Score)")
    print("=" * 60)

    # 8. Save Highest Performing Model to API
    api_dir = pathlib.Path("API/models")
    api_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = api_dir / "fraud_model.pkl"
    joblib.dump(models[winner], model_path)
    
    stats_path = api_dir / "feature_stats.pkl"
    joblib.dump(train_stats, stats_path)
    
    with open(api_dir / "winner.txt", "w") as f:
        f.write(winner)
        
    logger.info(f"Winner '{winner}' model saved to {model_path}")
    logger.info("Pipeline complete! Ready for Frontend & API deployment.")

if __name__ == "__main__":
    main()
