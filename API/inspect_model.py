import joblib

m = joblib.load("API/models/fraud_model.pkl")
stats = joblib.load("API/models/feature_stats.pkl")

print("XGBOOST FEATURES:", m.feature_names_in_)
print("\nSTATS KEYS:", stats.keys())
print("NUMERIC COLS:", stats.get("num_cols"))
print("CATEGORIES:", stats.get("categories"))
