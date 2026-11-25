"""
03_model.py
Feature engineering + fraud classification using Random Forest and manual SMOTE.
"""
import os
import time
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

# File loading logic
csv_filename = "transactions.csv" if os.path.exists(os.path.join(BASE_DIR, "transactions.csv")) else "transactions_sample.csv"
csv_path = os.path.join(BASE_DIR, csv_filename)
print(f"Loading data from: {csv_path}...")

df = pd.read_csv(csv_path, parse_dates=["timestamp"])
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# ---------------- Feature engineering ----------------
print("Engineering features...")
g = df.groupby("user_id")
df["prev_ts"] = g["timestamp"].shift(1)
df["prev_device"] = g["device_id"].shift(1)
df["prev_amount"] = g["amount"].shift(1)

df["minutes_since_last_txn"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60.0
df["minutes_since_last_txn"] = df["minutes_since_last_txn"].fillna(999999)

df["device_switch"] = ((df["device_id"] != df["prev_device"]) & df["prev_device"].notna()).astype(int)
df["fast_device_switch"] = ((df["device_switch"] == 1) & (df["minutes_since_last_txn"] < 10)).astype(int)

df["hour"] = df["timestamp"].dt.hour
df["is_odd_hour"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)
df["day_of_week"] = df["timestamp"].dt.dayofweek

cat_median = df.groupby("merchant_category")["amount"].transform("median")
df["amount_vs_category_median"] = df["amount"] / cat_median

df["amount_delta_from_prev"] = (df["amount"] - df["prev_amount"]).fillna(0)
df["high_velocity"] = (df["minutes_since_last_txn"] < 5).astype(int)

# Merchant-category historical fraud rate
cat_fraud_rate = df.groupby("merchant_category")["is_fraud"].transform("mean")
df["merchant_category_fraud_rate"] = cat_fraud_rate

le_method = LabelEncoder()
df["payment_method_enc"] = le_method.fit_transform(df["payment_method"])

feature_cols = [
    "amount", "minutes_since_last_txn", "device_switch", "fast_device_switch",
    "hour", "is_odd_hour", "day_of_week", "amount_vs_category_median",
    "amount_delta_from_prev", "high_velocity", "merchant_category_fraud_rate",
    "payment_method_enc"
]
X = df[feature_cols].fillna(0).values
y = df["is_fraud"].values

print(f"Full feature matrix: {X.shape}, fraud rate: {y.mean()*100:.3f}%")

# Subsampling logic for performance if running full 1.2M rows
SAMPLE_SIZE = 180_000
fraud_idx = np.where(y == 1)[0]
nonfraud_idx = np.where(y == 0)[0]

if len(df) > SAMPLE_SIZE:
    rng.shuffle(nonfraud_idx)
    n_nonfraud_sample = SAMPLE_SIZE - len(fraud_idx)
    sample_idx_all = np.concatenate([fraud_idx, nonfraud_idx[:n_nonfraud_sample]])
    rng.shuffle(sample_idx_all)
    X = X[sample_idx_all]
    y = y[sample_idx_all]
    print(f"Modeling subsample: {X.shape}, fraud rate: {y.mean()*100:.3f}%")

# ---------------- Train/validation/test split ----------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.35, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(0.25 / 0.35), random_state=42, stratify=y_temp
)
print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# ---------------- Manual SMOTE (Train set only) ----------------
def manual_smote(X_min, n_synthetic, k=5, seed=42):
    rng_local = np.random.default_rng(seed)
    k = min(k, max(1, X_min.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)
    sample_idx = rng_local.integers(0, X_min.shape[0], size=n_synthetic)
    neighbor_choice = rng_local.integers(1, k + 1, size=n_synthetic)
    neighbor_idx = idx[sample_idx, neighbor_choice]
    gaps = rng_local.random(size=(n_synthetic, 1))
    synthetic = X_min[sample_idx] + gaps * (X_min[neighbor_idx] - X_min[sample_idx])
    return synthetic

t0 = time.time()
X_train_min = X_train[y_train == 1]
X_train_maj = X_train[y_train == 0]
n_to_generate = max(0, X_train_maj.shape[0] - X_train_min.shape[0])

if n_to_generate > 0 and len(X_train_min) > 1:
    print(f"Generating {n_to_generate:,} synthetic minority samples via manual SMOTE...")
    X_synthetic = manual_smote(X_train_min, n_to_generate, k=5)
    X_train_res = np.vstack([X_train_maj, X_train_min, X_synthetic])
    y_train_res = np.concatenate([
        np.zeros(len(X_train_maj)), np.ones(len(X_train_min)), np.ones(len(X_synthetic))
    ])
    perm = rng.permutation(len(X_train_res))
    X_train_res, y_train_res = X_train_res[perm], y_train_res[perm]
else:
    X_train_res, y_train_res = X_train, y_train

print(f"SMOTE finished in {time.time()-t0:.1f}s. Resampled train set: {X_train_res.shape}")

# ---------------- Train Random Forest ----------------
print("Training Random Forest...")
t0 = time.time()
clf = RandomForestClassifier(
    n_estimators=120, max_depth=12, min_samples_leaf=3,
    n_jobs=-1, random_state=42
)
clf.fit(X_train_res, y_train_res)
print(f"Training done in {time.time()-t0:.1f}s")

# ---------------- Threshold tuning on Validation set ----------------
y_val_proba = clf.predict_proba(X_val)[:, 1]
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, y_val_proba)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_idx = np.argmax(f1_arr[:-1]) if len(thresh_arr) > 0 else 0
best_threshold = thresh_arr[best_idx] if len(thresh_arr) > 0 else 0.5
print(f"Best threshold: {best_threshold:.3f}")

# ---------------- Evaluation on Test set ----------------
y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= best_threshold).astype(int)

f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE (held-out test set)")
print("=" * 50)
print(f"F1-score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"Confusion matrix:\n{cm}")

importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(6).to_string())

# ---------------- Query latency check ----------------
db_path = os.path.join(BASE_DIR, "transactions.db")
avg_latency_ms = 0.0
if os.path.exists(db_path):
    con = sqlite3.connect(db_path)
    t0 = time.time()
    for _ in range(20):
        con.execute("""
            SELECT merchant_category, COUNT(*), AVG(amount)
            FROM transactions WHERE is_fraud = 1 GROUP BY merchant_category
        """).fetchall()
    avg_latency_ms = (time.time() - t0) / 20 * 1000
    con.close()
    print(f"\nAvg aggregate query latency: {avg_latency_ms:.1f} ms")

# Save model_results.txt
out_model_results = os.path.join(BASE_DIR, "model_results.txt")
with open(out_model_results, "w") as f:
    f.write(f"f1_score={f1:.4f}\n")
    f.write(f"roc_auc={roc_auc:.4f}\n")
    f.write(f"precision={precision:.4f}\n")
    f.write(f"recall={recall:.4f}\n")
    f.write(f"avg_query_latency_ms={avg_latency_ms:.1f}\n")
    f.write(f"train_rows={len(df):,}\n")
    f.write(f"fraud_rate_pct={y.mean()*100:.3f}\n")
    f.write("top_features=" + ", ".join(importances.head(4).index.tolist()) + "\n")

print(f"\nSaved {out_model_results}")