"""
05_advanced_rules_model.py
Implements the exact behavioral-rule logic described in the fraud detection
spec (velocity rules, device switching, card-testing/amount-anomaly rules),
adds them as features, and retrains the classifier.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)
t_start = time.time()

# ---------------- Load dataset ----------------
csv_filename = "transactions.csv" if os.path.exists(os.path.join(BASE_DIR, "transactions.csv")) else "transactions_sample.csv"
csv_path = os.path.join(BASE_DIR, csv_filename)
print(f"Loading data from: {csv_path}...")
df = pd.read_csv(csv_path, parse_dates=["timestamp"])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ---------------- Base temporal features ----------------
print("Extracting temporal and behavioral features...")
g = df.groupby("user_id")
df["prev_ts"] = g["timestamp"].shift(1)
df["prev_device"] = g["device_id"].shift(1)
df["prev_amount"] = g["amount"].shift(1)
df["seconds_since_last_txn"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()
df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(999999)
df["device_switched"] = ((df["device_id"] != df["prev_device"]) & df["prev_device"].notna()).astype(int)

df["hour"] = df["timestamp"].dt.hour
df["is_odd_hour"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
df["day_of_week"] = df["timestamp"].dt.dayofweek

# ---------------- User historical average amount ----------------
grp_amt = df.groupby("user_id")["amount"]
cumsum_prev = grp_amt.cumsum() - df["amount"]
cumcount = grp_amt.cumcount()
df["user_hist_avg_amount"] = cumsum_prev / cumcount.replace(0, np.nan)
df["user_hist_avg_amount"] = df["user_hist_avg_amount"].fillna(df["amount"])
df["amount_to_avg_ratio"] = df["amount"] / (df["user_hist_avg_amount"] + 1e-5)

# ---------------- A. Velocity rule ----------------
high_value_cutoff = df["amount"].quantile(0.90)
df["velocity_risk_flag"] = (
    (df["seconds_since_last_txn"] < 30) & (df["amount"] > high_value_cutoff)
).astype(int)

# ---------------- B. Device switching rule ----------------
df["device_mismatch_flag"] = (
    (df["device_switched"] == 1) & (df["seconds_since_last_txn"] < 300)
).astype(int)

# ---------------- C. Card-testing rule ----------------
micro_cutoff = df["amount"].quantile(0.02)
df["card_testing_flag"] = (
    (df["prev_amount"] < micro_cutoff)
    & (df["amount"] > 5 * df["user_hist_avg_amount"])
    & (df["seconds_since_last_txn"] < 600)
).astype(int)

print(f"Velocity flags: {df['velocity_risk_flag'].sum():,} ({df['velocity_risk_flag'].mean()*100:.3f}% of txns)")
print(f"Device mismatch flags: {df['device_mismatch_flag'].sum():,} ({df['device_mismatch_flag'].mean()*100:.3f}% of txns)")
print(f"Card-testing flags: {df['card_testing_flag'].sum():,} ({df['card_testing_flag'].mean()*100:.3f}% of txns)")

for col in ["velocity_risk_flag", "device_mismatch_flag", "card_testing_flag"]:
    sub = df[df[col] == 1]
    if len(sub) > 0:
        print(f"  Fraud rate when {col}=1: {sub['is_fraud'].mean()*100:.2f}% (n={len(sub):,}) vs baseline {df['is_fraud'].mean()*100:.2f}%")

cat_median = df.groupby("merchant_category")["amount"].transform("median")
df["amount_vs_category_median"] = df["amount"] / cat_median
cat_fraud_rate = df.groupby("merchant_category")["is_fraud"].transform("mean")
df["merchant_category_fraud_rate"] = cat_fraud_rate
le_method = LabelEncoder()
df["payment_method_enc"] = le_method.fit_transform(df["payment_method"])

feature_cols = [
    "amount", "seconds_since_last_txn", "amount_to_avg_ratio",
    "velocity_risk_flag", "device_mismatch_flag", "card_testing_flag",
    "device_switched", "hour", "is_odd_hour", "day_of_week",
    "amount_vs_category_median", "merchant_category_fraud_rate", "payment_method_enc"
]
X_full = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
y_full = df["is_fraud"].values

# ---------------- Stratified subsample ----------------
SAMPLE_SIZE = 180_000
fraud_idx = np.where(y_full == 1)[0]
nonfraud_idx = np.where(y_full == 0)[0]

if len(df) > SAMPLE_SIZE:
    rng.shuffle(nonfraud_idx)
    n_nonfraud_sample = SAMPLE_SIZE - len(fraud_idx)
    sample_idx_all = np.concatenate([fraud_idx, nonfraud_idx[:n_nonfraud_sample]])
    rng.shuffle(sample_idx_all)
    X = X_full[sample_idx_all]
    y = y_full[sample_idx_all]
    print(f"\nModeling subsample: {X.shape}, fraud rate: {y.mean()*100:.3f}%")
else:
    X = X_full
    y = y_full

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(0.25 / 0.35), random_state=42, stratify=y_temp)
print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

def manual_smote(X_min, n_synthetic, k=5, seed=42):
    rng_local = np.random.default_rng(seed)
    k = min(k, max(1, X_min.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    sample_idx = rng_local.integers(0, X_min.shape[0], size=n_synthetic)
    neighbor_choice = rng_local.integers(1, k + 1, size=n_synthetic)
    neighbor_idx = idx[sample_idx, neighbor_choice]
    gaps = rng_local.random(size=(n_synthetic, 1))
    return X_min[sample_idx] + gaps * (X_min[neighbor_idx] - X_min[sample_idx])

X_train_min = X_train[y_train == 1]
X_train_maj = X_train[y_train == 0]
n_to_generate = max(0, X_train_maj.shape[0] - X_train_min.shape[0])

if n_to_generate > 0 and len(X_train_min) > 1:
    X_synth = manual_smote(X_train_min, n_to_generate, k=5)
    X_train_res = np.vstack([X_train_maj, X_train_min, X_synth])
    y_train_res = np.concatenate([np.zeros(len(X_train_maj)), np.ones(len(X_train_min)), np.ones(len(X_synth))])
    perm = rng.permutation(len(X_train_res))
    X_train_res, y_train_res = X_train_res[perm], y_train_res[perm]
else:
    X_train_res, y_train_res = X_train, y_train

print("Training Random Forest with rule-based features added...")
t0 = time.time()
clf = RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42)
clf.fit(X_train_res, y_train_res)
print(f"Trained in {time.time()-t0:.1f}s")

y_val_proba = clf.predict_proba(X_val)[:, 1]
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, y_val_proba)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_idx = np.argmax(f1_arr[:-1]) if len(thresh_arr) > 0 else 0
best_threshold = thresh_arr[best_idx] if len(thresh_arr) > 0 else 0.5

y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 55)
print("UPDATED MODEL PERFORMANCE (with rule-based features)")
print("=" * 55)
print(f"Threshold: {best_threshold:.3f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.to_string())

# Risk bucketing on the FULL dataset
full_proba = clf.predict_proba(X_full)[:, 1]
risk_bucket = np.where(full_proba > 0.75, "HIGH", np.where(full_proba >= 0.30, "MEDIUM", "LOW"))
bucket_counts = pd.Series(risk_bucket).value_counts()
print("\nRisk bucket distribution across all transactions:")
print(bucket_counts.to_string())
bucket_fraud_rate = pd.DataFrame({"bucket": risk_bucket, "is_fraud": y_full}).groupby("bucket")["is_fraud"].mean() * 100
print("\nActual fraud rate within each risk bucket:")
print(bucket_fraud_rate.round(2).to_string())

out_model_v2 = os.path.join(BASE_DIR, "model_results_v2.txt")
with open(out_model_v2, "w") as f:
    f.write(f"f1_score={f1:.4f}\nroc_auc={roc_auc:.4f}\nprecision={precision:.4f}\nrecall={recall:.4f}\n")
    f.write(f"threshold={best_threshold:.3f}\n")
    f.write("top_features=" + ", ".join(importances.head(5).index.tolist()) + "\n")
    for b in ["LOW", "MEDIUM", "HIGH"]:
        cnt = bucket_counts.get(b, 0)
        fr = bucket_fraud_rate.get(b, 0.0)
        f.write(f"bucket_{b}_count={cnt}\nbucket_{b}_fraud_rate_pct={fr:.2f}\n")

print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
print(f"Saved {out_model_v2}")