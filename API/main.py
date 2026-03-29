from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
import os
import uuid
import time

app = FastAPI(title="Credit Card Fraud Engine API - Antigravity")

# Allow frontend HTML/JS files to communicate with the local API securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the finalized WGAN-GP XGBoost model & feature scaler parameters
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fraud_model.pkl")
STATS_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_stats.pkl")

print("Initializing Enterprise ML Pipeline...")
model = joblib.load(MODEL_PATH)
stats = joblib.load(STATS_PATH)
categories = list(stats['categories'])
scaler = stats['scaler']

# 2. In-Memory "Mock Redis" Database for extreme real-time latency
# Format: user_id -> [{"timestamp": datetime, "amount": float}]
user_history = {}

# 3. Pydantic Scheme Definition
class TransactionRequest(BaseModel):
    user_id: str
    merchant_id: str
    amount: float = Field(..., gt=0)
    category: str
    dist_to_merch: float = Field(-1.0, description="Distance in miles")
    timestamp: str = Field(None, description="ISO format. If missing, assumes current UTC time.")

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    trigger_threshold: float
    latency_ms: float

@app.post("/predict", response_model=PredictionResponse)
def execute_prediction(txn: TransactionRequest):
    start_time = time.time()
    
    # ---------------------------------------------------------
    # A. Parse Defaults
    # ---------------------------------------------------------
    if not txn.timestamp:
        txn_time = datetime.utcnow()
    else:
        try:
            # Handle standard ISO string
            txn_time = pd.to_datetime(txn.timestamp).to_pydatetime()
            if txn_time.tzinfo is not None:
                txn_time = txn_time.replace(tzinfo=None)
        except Exception:
            raise HTTPException(400, "Invalid ISO timestamp format")

    dist = txn.dist_to_merch if txn.dist_to_merch >= 0 else stats.get("dist_median", 0.0)
    
    # ---------------------------------------------------------
    # B. Mock Query to Redis for User Behavioral State
    # ---------------------------------------------------------
    uid = txn.user_id
    if uid not in user_history:
        user_history[uid] = []
    
    history = user_history[uid]
    
    time_since_last = -1.0
    velocity_24h = 0.0
    spend_24h = 0.0
    
    if history:
        history.sort(key=lambda x: x["timestamp"]) # Ensure chronological consistency
        last_txn = history[-1]
        time_since_last = (txn_time - last_txn["timestamp"]).total_seconds()
        
        # 24hr lookback window implementation
        cutoff = txn_time - timedelta(hours=24)
        for h in reversed(history):
            if h["timestamp"] > cutoff:
                velocity_24h += 1
                spend_24h += h["amount"]
            else:
                break
                
    # Append the current transaction to history AFTER compiling rolling history
    # This matches the behavior of 'closed="left"' in Pandas rolling windows
    history.append({"timestamp": txn_time, "amount": txn.amount})
    
    # ---------------------------------------------------------
    # C. Engineer Universal Features On-the-Fly
    # ---------------------------------------------------------
    hour = txn_time.hour + txn_time.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    dow = txn_time.weekday()
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    
    amt_log = np.log1p(txn.amount)
    mean_spend = stats.get("user_avg", {}).get(uid, stats.get("global_avg", 0.0))
    amt_vs_avg = txn.amount / (mean_spend + 1e-5)
    
    # Strictly map the category
    try:
        cat_code = float(categories.index(txn.category))
    except ValueError:
        cat_code = -1.0 # Unknown unseen category
        
    # Scale exactly to the N(0,1) Quantile distribution fit during training
    # Order matched closely to src/features.py numeric definitions
    num_arr = np.array([[txn.amount, amt_log, amt_vs_avg, velocity_24h, dist, spend_24h, time_since_last]])
    num_scaled = scaler.transform(num_arr)[0]
    
    # ---------------------------------------------------------
    # D. Inference & Output Allocation
    # ---------------------------------------------------------
    # Build payload ensuring exact feature positioning for the XGBoost engine
    features = {
        "amount": num_scaled[0],
        "category": cat_code,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "amount_log": num_scaled[1],
        "dist_to_merch": num_scaled[4],
        "time_since_last_txn": num_scaled[6],
        "velocity_24h": num_scaled[3],
        "spend_24h": num_scaled[5],
        "amount_vs_avg": num_scaled[2]
    }
    
    df_payload = pd.DataFrame([features])
    
    # Run absolute probability matrix
    pred_prob = model.predict_proba(df_payload)[0][1]
    
    # The optimal F1 threshold discovered automatically in Phase 9
    THRESHOLD = 0.35
    is_fraud = bool(pred_prob > THRESHOLD)
    
    # End timing metric
    end_time = time.time()
    latency = round((end_time - start_time) * 1000, 2)
    
    return PredictionResponse(
        transaction_id=str(uuid.uuid4()),
        is_fraud=is_fraud,
        fraud_probability=round(float(pred_prob), 4),
        trigger_threshold=THRESHOLD,
        latency_ms=latency
    )

# ---------------------------------------------------------
# Web Server Hosting (For Cloud Deployment)
# ---------------------------------------------------------
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

@app.get("/")
def load_dashboard():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/style.css")
def load_css():
    return FileResponse(os.path.join(frontend_path, "style.css"))

@app.get("/app.js")
def load_script():
    return FileResponse(os.path.join(frontend_path, "app.js"))