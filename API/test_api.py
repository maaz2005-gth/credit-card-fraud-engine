import requests
import time
import json

url = "http://127.0.0.1:8000/predict"

print("="*60)
print("--- TRANSACTION 1: Normal Grocery Shop ---")
print("="*60)
t1 = {
  "user_id": "cust_123",
  "merchant_id": "merch_999",
  "amount": 42.50,
  "category": "grocery_pos",
  "dist_to_merch": 2.5
}
r1 = requests.post(url, json=t1)
print(json.dumps(r1.json(), indent=2))

print("\n[Waiting 2 seconds to simulate a scammer stealing the card...]\n")
time.sleep(2)

print("="*60)
print("--- TRANSACTION 2: Fraudulent Back-to-Back High-Value Spend ---")
print("="*60)
t2 = {
  "user_id": "cust_123",
  "merchant_id": "merch_555",
  "amount": 850.00,
  "category": "entertainment",
  "dist_to_merch": 105.0
}
r2 = requests.post(url, json=t2)
print(json.dumps(r2.json(), indent=2))
