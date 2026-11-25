"""
01_generate_data.py
Generates a synthetic (simulated) real-time digital payment transaction dataset.
This is clearly synthetic data, generated for portfolio/learning purposes -
not real transaction records.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Dynamically set output directory to current folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(42)

N = 1_200_000  # 1.2M+ transactions
N_USERS = 60_000
N_MERCHANTS = 2_500

MERCHANT_CATEGORIES = [
    "Electronics", "Grocery", "Travel", "Gaming", "Gift Cards",
    "Fashion", "Food Delivery", "Utilities", "Crypto Exchange", "Subscriptions"
]

CATEGORY_BASE_FRAUD_RATE = {
    "Electronics": 0.015, "Grocery": 0.001, "Travel": 0.010, "Gaming": 0.022,
    "Gift Cards": 0.035, "Fashion": 0.004, "Food Delivery": 0.002,
    "Utilities": 0.0007, "Crypto Exchange": 0.040, "Subscriptions": 0.003
}
PAYMENT_METHODS = ["card", "upi", "wallet", "netbanking"]

print(f"Generating {N:,} synthetic transactions...")

# --- Core entities ---
user_ids = rng.integers(0, N_USERS, size=N)
merchant_ids = rng.integers(0, N_MERCHANTS, size=N)
merchant_categories = np.array(MERCHANT_CATEGORIES)[rng.integers(0, len(MERCHANT_CATEGORIES), size=N)]

device_ids = rng.integers(0, N, size=N) % (N_USERS * 2)

# Timestamps across a 12-month window
start = datetime(2025, 4, 1)
day_offsets = rng.integers(0, 365, size=N)
hour_weights = np.concatenate([np.full(6, 0.3), np.full(10, 1.0), np.full(6, 1.6), np.full(2, 0.5)])
hours = rng.choice(np.arange(24), size=N, p=hour_weights / hour_weights.sum())
minutes = rng.integers(0, 60, size=N)
timestamps = [start + timedelta(days=int(d), hours=int(h), minutes=int(m))
              for d, h, m in zip(day_offsets, hours, minutes)]

# Transaction amount
category_scale = {
    "Electronics": 180, "Grocery": 25, "Travel": 350, "Gaming": 40,
    "Gift Cards": 60, "Fashion": 55, "Food Delivery": 18,
    "Utilities": 45, "Crypto Exchange": 500, "Subscriptions": 15
}
scales = np.array([category_scale[c] for c in merchant_categories])
amounts = rng.lognormal(mean=0, sigma=0.9, size=N) * scales
amounts = np.round(amounts, 2)

payment_methods = np.array(PAYMENT_METHODS)[rng.integers(0, len(PAYMENT_METHODS), size=N)]

df = pd.DataFrame({
    "transaction_id": np.arange(1, N + 1),
    "timestamp": timestamps,
    "user_id": user_ids,
    "device_id": device_ids,
    "merchant_id": merchant_ids,
    "merchant_category": merchant_categories,
    "payment_method": payment_methods,
    "amount": amounts,
})
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# --- Inject realistic fraud signal ---
# 1. Category base rate
base_fraud_prob = df["merchant_category"].map(CATEGORY_BASE_FRAUD_RATE).values.copy()

# 2. Odd-hour transactions (12am-5am)
hour_of_txn = pd.to_datetime(df["timestamp"]).dt.hour.values
odd_hour = (hour_of_txn >= 0) & (hour_of_txn <= 5)
base_fraud_prob = base_fraud_prob + odd_hour * 0.02

# 3. High amount relative to category median
cat_median = df.groupby("merchant_category")["amount"].transform("median").values
high_amount = (df["amount"].values > cat_median * 4)
base_fraud_prob = base_fraud_prob + high_amount * 0.03

# 4. Fast device switching (<10 min)
df["prev_device"] = df.groupby("user_id")["device_id"].shift(1)
df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)
time_delta_min = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["prev_ts"])).dt.total_seconds() / 60.0
device_switch_fast = (df["device_id"] != df["prev_device"]) & (time_delta_min < 10) & df["prev_device"].notna()
base_fraud_prob = base_fraud_prob + device_switch_fast.values * 0.15

# 5. Velocity risk (<5 min gap)
rolling_gap_small = (time_delta_min < 5) & df["prev_ts"].notna()
base_fraud_prob = base_fraud_prob + rolling_gap_small.values * 0.05

base_fraud_prob = np.clip(base_fraud_prob, 0, 0.9)
is_fraud = rng.binomial(1, base_fraud_prob)
df["is_fraud"] = is_fraud

# Transaction status and chargebacks
organic_fail_prob = 0.04
fraud_fail_prob = 0.35
fail_prob = np.where(df["is_fraud"] == 1, fraud_fail_prob, organic_fail_prob)
failed = rng.binomial(1, fail_prob)
df["transaction_status"] = np.where(failed == 1, "failed", "success")
df["chargeback"] = ((df["is_fraud"] == 1) & (df["transaction_status"] == "success")).astype(int)

df = df.drop(columns=["prev_device", "prev_ts"])

# --- Save output datasets ---
out_path = os.path.join(BASE_DIR, "transactions.csv")
sample_path = os.path.join(BASE_DIR, "transactions_sample.csv")

print("Writing full dataset to disk (this may take 10-20 seconds)...")
df.to_csv(out_path, index=False)
df.head(5000).to_csv(sample_path, index=False)

print(f"Rows generated: {len(df):,}")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%")
print(f"Chargeback rate (of all txns): {df['chargeback'].mean()*100:.3f}%")
print(f"Failed txn rate: {(df['transaction_status']=='failed').mean()*100:.2f}%")
print(f"Saved full data to: {out_path}")
print(f"Saved 5k sample to: {sample_path}")