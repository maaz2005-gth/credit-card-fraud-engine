import pygad
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from features import engineer_features
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 1. LOAD AND PREP DATA
logger.info("Loading Sparkov Dataset for Genetic Evolution...")
df_raw = pd.read_csv("data/harmonized.csv", low_memory=False)
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")
df_raw = df_raw[df_raw["source_domain"] == "sparkov_us_pos"].copy()

# Crucial: Genetic Algorithms over 1 Million rows with 40 generations = 50 hours of compute.
# We take a high-fidelity 5% stratified random sample to accomplish this in seconds.
logger.info("Subsampling 5% of data to accelerate evolution...")
df_sample, _ = train_test_split(df_raw, train_size=0.05, stratify=df_raw["is_fraud"], random_state=42)

df_processed, y_processed, _, _ = engineer_features(df_sample, fit=True)
feature_names = df_processed.columns.tolist()

# Fast Numpy representations for GA slicing
X = df_processed.values
y = y_processed.values
scale_pos = (y == 0).sum() / max(1, (y == 1).sum())

# 2. THE FITNESS FUNCTION (F1 Score cross-validation)
def fitness_func(ga_instance, solution, solution_idx):
    # Select features where the 'gene' is 1
    selected_indices = [i for i, bit in enumerate(solution) if bit == 1]
    
    if len(selected_indices) == 0:
        return 0.0 # If GA tries deleting all features, fail it.
    
    X_subset = X[:, selected_indices]
    
    # Train an aggressively fast version of XGBoost
    model = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    # Evaluate using F1-score (Crucial for fraud!)
    scores = cross_val_score(model, X_subset, y, cv=3, scoring='f1')
    return np.mean(scores)

# 3. RUN GA (10 Generations)
logger.info(f"Starting Genetic Algorithm: {len(feature_names)} Starting Genes.")
ga_instance = pygad.GA(
    num_generations=15,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=10, 
    num_genes=len(feature_names),
    gene_type=int,
    gene_space=[0, 1]
)

ga_instance.run()

# 4. RESULTS
solution, fitness, idx = ga_instance.best_solution()
winner_features = [feature_names[i] for i, bit in enumerate(solution) if bit == 1]

print("\n" + "="*60)
print(f"EVOLUTION COMPLETE")
print("="*60)
print(f"Highest F1 Score Found: {fitness:.4f}")
print(f"Winning Subset of Features ({len(winner_features)} features):")
for f in winner_features:
    print(f" - {f}")

# Save the winner names
os.makedirs("API/models", exist_ok=True)
joblib.dump(winner_features, 'API/models/ga_best_features.pkl')
print("\n🧬 Saved winning features to API/models/ga_best_features.pkl")
