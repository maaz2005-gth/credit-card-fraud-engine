"""
data_harmonizer.py
==================
Ingests multiple fraud datasets and maps them into a unified Behavioral Master Schema.

Master Schema (column -> description):
    timestamp       : UTC transaction datetime
    user_id         : Anonymized user/card identifier (str)
    amount          : Transaction amount in native currency (float)
    merchant_id     : Merchant name/code/email (str, or 'UNKNOWN')
    category        : Transaction category (str, or 'UNKNOWN')
    hour_sin        : Sine encoding of transaction hour (float)
    hour_cos        : Cosine encoding of transaction hour (float)
    dow_sin         : Sine encoding of day-of-week (float)
    dow_cos         : Cosine encoding of day-of-week (float)
    amount_log      : log1p(amount) for skew correction (float)
    dist_to_merch   : Distance user->merchant (float, NaN if unavailable)
    is_fraud        : Ground-truth label (int, 0 or 1)
    source_domain   : Dataset origin tag — used for LODO cross-validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MASTER_COLS = [
    "timestamp", "user_id", "amount", "merchant_id", "category",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "amount_log", "dist_to_merch", "is_fraud", "source_domain"
]

def _cyclical(values: pd.Series, period: float):
    rad = 2 * np.pi * values / period
    return np.sin(rad), np.cos(rad)

class PandasHarmonizer:
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)

    def build(self) -> pd.DataFrame:
        frames = []
        
        sparkov_path = self.data_dir / "credit_card_fraud.csv"
        if sparkov_path.exists():
            logger.info("Loading Sparkov (Physical POS) dataset …")
            frames.append(self._harmonize_sparkov(sparkov_path))
            
        ieee_path = self.data_dir / "train_transaction.csv"
        if ieee_path.exists():
            logger.info("Loading IEEE-CIS (E-commerce) dataset …")
            frames.append(self._harmonize_ieee(ieee_path))
            
        if not frames:
            raise FileNotFoundError("No datasets found in: " + str(self.data_dir))
            
        merged = pd.concat(frames, ignore_index=True)
        merged = merged[MASTER_COLS]
        self._validate(merged)
        
        logger.info(
            "Harmonized dataset: %d rows | domains: %s | fraud rate: %.4f%%",
            len(merged),
            merged["source_domain"].value_counts().to_dict(),
            merged["is_fraud"].mean() * 100,
        )
        return merged

    def _harmonize_sparkov(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, low_memory=False)
        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(raw["trans_date_trans_time"], errors="coerce")
        hour = out["timestamp"].dt.hour.fillna(0)
        dow  = out["timestamp"].dt.dayofweek.fillna(0)
        out["hour_sin"], out["hour_cos"] = _cyclical(hour, 24)
        out["dow_sin"],  out["dow_cos"]  = _cyclical(dow,  7)
        out["user_id"]     = raw["cc_num"].astype(str)
        out["amount"]      = raw["amt"].astype(float)
        out["merchant_id"] = raw["merchant"].astype(str)
        out["category"]    = raw["category"].astype(str)
        out["amount_log"]  = np.log1p(out["amount"])
        out["dist_to_merch"] = np.sqrt((raw["lat"] - raw["merch_lat"]) ** 2 + (raw["long"] - raw["merch_long"]) ** 2)
        out["is_fraud"]      = raw["is_fraud"].astype(int)
        out["source_domain"] = "sparkov_us_pos"
        return out

    def _harmonize_ieee(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path, low_memory=False)
        out = pd.DataFrame()
        
        # TransactionDT is seconds from an arbitrary date. We assume 2017-12-01.
        base_date = datetime.datetime(2017, 12, 1)
        out["timestamp"] = base_date + pd.to_timedelta(raw["TransactionDT"], unit="s")
        hour = out["timestamp"].dt.hour.fillna(0)
        dow  = out["timestamp"].dt.dayofweek.fillna(0)
        out["hour_sin"], out["hour_cos"] = _cyclical(hour, 24)
        out["dow_sin"],  out["dow_cos"]  = _cyclical(dow,  7)
        
        # Construct proxy user_id from card identifiers and address
        out["user_id"] = (raw["card1"].fillna(0).astype(str) + "_" + 
                          raw["card2"].fillna(0).astype(str) + "_" + 
                          raw["addr1"].fillna(0).astype(str))
                          
        out["amount"]      = raw["TransactionAmt"].astype(float)
        out["merchant_id"] = raw["P_emaildomain"].fillna("UNKNOWN").astype(str)
        out["category"]    = raw["ProductCD"].astype(str)
        out["amount_log"]  = np.log1p(out["amount"])
        out["dist_to_merch"] = raw["dist1"].astype(float)
        
        out["is_fraud"]      = raw["isFraud"].astype(int)
        out["source_domain"] = "ieee_cis_ecommerce"
        return out

    def _validate(self, df: pd.DataFrame):
        missing = [c for c in MASTER_COLS if c not in df.columns]
        if missing: raise ValueError(f"Missing cols: {missing}")
        logger.info("  Schema validation passed ✓")
        
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    h = PandasHarmonizer(data_dir=data_dir)
    harmonized = h.build()
    
    out_path = Path(data_dir) / "harmonized.csv"
    harmonized.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)
