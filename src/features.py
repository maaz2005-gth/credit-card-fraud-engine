import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, fit: bool = True, stats: dict = None):
    """
    Apply universal feature engineering to a harmonized dataset.
    Extracts underlying behavioral signals regardless of whether the transaction 
    happened at a physical grocery store or an internet marketplace.
    """
    df = df.copy()

    # ── 1. Velocity ──────────────────────────────────────────────────────────
    df = _add_velocity_24h(df)

    # ── 2. Amount vs. user historical average ────────────────────────────────
    if fit:
        stats = stats or {}
        user_avg = df.groupby("user_id")["amount"].mean().rename("_user_avg")
        stats["user_avg"] = user_avg
        global_avg = df["amount"].mean()
        stats["global_avg"] = global_avg
    else:
        user_avg = stats["user_avg"]
        global_avg = stats["global_avg"]

    df["amount_vs_avg"] = (
        df["amount"]
        .div(df["user_id"].map(user_avg).fillna(global_avg))
        .replace([np.inf, -np.inf], 0)
        .fillna(1.0)
    )

    # ── 3. Imputation & Strict Distribution Alignment ────────────────────────
    if fit:
        dist_median = df["dist_to_merch"].median()
        stats["dist_median"] = float(dist_median) if not np.isnan(dist_median) else -1.0
    df["dist_to_merch"] = df["dist_to_merch"].fillna(stats.get("dist_median", -1.0))
    df["velocity_24h"] = df["velocity_24h"].fillna(0.0)
    if "spend_24h" in df.columns:
        df["spend_24h"] = df["spend_24h"].fillna(0.0)

    # Force numeric features into strict Gaussian distributions to erase domain shift
    num_cols = ["amount", "amount_log", "amount_vs_avg", "velocity_24h", "dist_to_merch", "spend_24h", "time_since_last_txn"]
    if fit:
        # subsample limits memory usage on huge datasets while calculating quantiles
        scaler = QuantileTransformer(output_distribution="normal", random_state=42, n_quantiles=1000, copy=False)
        df[num_cols] = scaler.fit_transform(df[num_cols])
        stats["scaler"] = scaler
    else:
        df[num_cols] = stats["scaler"].transform(df[num_cols])

    # ── 4. Split and Return ──────────────────────────────────────────────────
    y = df["is_fraud"].astype(int)
    domain = df["source_domain"]
    
    # Encode categorical text feature strictly
    if "category" in df.columns:
        if fit:
            stats["categories"] = df["category"].unique()
        df["category"] = pd.Categorical(df["category"], categories=stats.get("categories", []))
        df["category"] = df["category"].cat.codes.astype(float)
    
    drop = ["timestamp", "user_id", "merchant_id", "is_fraud", "source_domain"]
    X = df.drop(columns=[c for c in drop if c in df.columns])

    logger.info(
        "Feature matrix: %d rows × %d features | fraud=%.2f%%",
        len(X), X.shape[1], y.mean() * 100,
    )
    return X, y, domain, stats

def _add_velocity_24h(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates velocity, rolling spend, and time deltas."""
    df = df.sort_values(["user_id", "timestamp"])
    
    # 1. Time since last transaction
    df["time_since_last_txn"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(-1.0)
    
    # 2. Fast window calculation for 24h metrics
    def group_metrics(group):
        timestamps = group["timestamp"].values
        amounts = group["amount"].values
        n = len(timestamps)
        vel = np.zeros(n, dtype=float)
        spend = np.zeros(n, dtype=float)
        
        for i in range(n):
            ts = timestamps[i]
            cutoff = ts - np.timedelta64(24, "h")
            j = i - 1
            v = 0
            s = 0.0
            # Look backwards up to 24h
            while j >= 0 and timestamps[j] > cutoff:
                v += 1
                s += amounts[j]
                j -= 1
            vel[i] = v
            spend[i] = s
        return pd.DataFrame({"velocity_24h": vel, "spend_24h": spend}, index=group.index)
        
    res = df.groupby("user_id", group_keys=False).apply(group_metrics, include_groups=False)
    df["velocity_24h"] = res["velocity_24h"]
    df["spend_24h"] = res["spend_24h"]
    return df