"""
02_sql_analysis.py
Loads the transactions into SQLite and runs SQL analysis
(CTEs, window functions - LAG/LEAD, DENSE_RANK) to surface business insights.
"""
import os
import sqlite3
import pandas as pd

# Dynamically resolve directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for full dataset or fallback to sample
csv_filename = "transactions.csv" if os.path.exists(os.path.join(BASE_DIR, "transactions.csv")) else "transactions_sample.csv"
csv_path = os.path.join(BASE_DIR, csv_filename)
print(f"Loading data from: {csv_path}")

df = pd.read_csv(csv_path, parse_dates=["timestamp"])

# Setup SQLite database
db_path = os.path.join(BASE_DIR, "transactions.db")
con = sqlite3.connect(db_path)
df.to_sql("transactions", con, if_exists="replace", index=False)
con.execute("CREATE INDEX IF NOT EXISTS idx_user ON transactions(user_id, timestamp);")
con.execute("CREATE INDEX IF NOT EXISTS idx_merchant ON transactions(merchant_category);")
con.commit()

print("=" * 70)
print("QUERY 1: Merchant categories driving chargeback losses (window fn: DENSE_RANK)")
print("=" * 70)
q1 = """
WITH category_stats AS (
    SELECT
        merchant_category,
        COUNT(*) AS total_txns,
        SUM(chargeback) AS chargeback_count,
        ROUND(100.0 * SUM(chargeback) / COUNT(*), 3) AS chargeback_rate_pct,
        ROUND(SUM(CASE WHEN chargeback = 1 THEN amount ELSE 0 END), 2) AS chargeback_amount
    FROM transactions
    GROUP BY merchant_category
),
ranked AS (
    SELECT *,
        DENSE_RANK() OVER (ORDER BY chargeback_amount DESC) AS risk_rank
    FROM category_stats
)
SELECT * FROM ranked ORDER BY risk_rank LIMIT 10;
"""
cat_df = pd.read_sql(q1, con)
print(cat_df.to_string(index=False))

total_chargeback_amt = cat_df["chargeback_amount"].sum()
top3_amt = cat_df.sort_values("chargeback_amount", ascending=False).head(3)["chargeback_amount"].sum()

# Recompute total across ALL categories for an accurate percentage
full_total_res = pd.read_sql("SELECT SUM(CASE WHEN chargeback=1 THEN amount ELSE 0 END) AS t FROM transactions", con)["t"][0]
full_total = full_total_res if full_total_res and full_total_res > 0 else 1.0
top3_share = 100 * top3_amt / full_total
print(f"\n>> Top 3 riskiest merchant categories account for {top3_share:.1f}% of total chargeback value.")

print("\n" + "=" * 70)
print("QUERY 2: Time-delta between a user's consecutive transactions (window fn: LAG)")
print("=" * 70)
q2 = """
WITH ordered AS (
    SELECT
        user_id, transaction_id, timestamp, device_id, is_fraud,
        LAG(timestamp) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_ts,
        LAG(device_id) OVER (PARTITION BY user_id ORDER BY timestamp) AS prev_device
    FROM transactions
),
deltas AS (
    SELECT *,
        (JULIANDAY(timestamp) - JULIANDAY(prev_ts)) * 24 * 60 AS minutes_since_last_txn
    FROM ordered
    WHERE prev_ts IS NOT NULL
)
SELECT
    CASE WHEN minutes_since_last_txn < 5 THEN '<5 min'
         WHEN minutes_since_last_txn < 30 THEN '5-30 min'
         WHEN minutes_since_last_txn < 1440 THEN '30min-1day'
         ELSE '>1 day' END AS gap_bucket,
    COUNT(*) AS txn_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM deltas
GROUP BY gap_bucket
ORDER BY fraud_rate_pct DESC;
"""
gap_df = pd.read_sql(q2, con)
print(gap_df.to_string(index=False))

print("\n" + "=" * 70)
print("QUERY 3: Payment success rate trend by month (business KPI)")
print("=" * 70)
q3 = """
SELECT
    strftime('%Y-%m', timestamp) AS month,
    COUNT(*) AS total_txns,
    ROUND(100.0 * SUM(CASE WHEN transaction_status='success' THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct
FROM transactions
GROUP BY month
ORDER BY month;
"""
trend_df = pd.read_sql(q3, con)
print(trend_df.to_string(index=False))

# Save output CSVs and summary findings in the project folder
cat_df.to_csv(os.path.join(BASE_DIR, "output_category_risk.csv"), index=False)
gap_df.to_csv(os.path.join(BASE_DIR, "output_velocity_risk.csv"), index=False)
trend_df.to_csv(os.path.join(BASE_DIR, "output_success_trend.csv"), index=False)

with open(os.path.join(BASE_DIR, "sql_findings.txt"), "w") as f:
    f.write(f"top3_chargeback_share_pct={top3_share:.1f}\n")
    top_cats = cat_df.sort_values("chargeback_amount", ascending=False).head(3)["merchant_category"].tolist()
    f.write(f"top3_categories={top_cats}\n")

con.close()
print("\nSaved: output_category_risk.csv, output_velocity_risk.csv, output_success_trend.csv, sql_findings.txt")