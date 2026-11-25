# Real-Time Payment Transaction Risk & Fraud Analytics Pipeline

An end-to-end data pipeline and analytics project simulating a fintech payment
processor's risk operations: transaction generation → SQL-based risk analysis →
fraud classification model → executive dashboard.

**Note on data:** the transaction dataset is synthetically generated (see
`01_generate_data.py`) with realistic, documented fraud-injection logic
(risky merchant categories, odd-hour activity, fast device switching,
transaction velocity). It is not real financial data — this is a portfolio /
learning project, and every number in this README comes from actually running
the code in this repo.

## Pipeline

1. **`01_generate_data.py`** — Generates 1,200,000 synthetic transactions
   across 60,000 users, 2,500 merchants, and 10 merchant categories over a
   12-month window, with realistic fraud, chargeback, and failure patterns.
2. **`02_sql_analysis.py`** — Loads the data into SQLite and runs SQL using
   CTEs and window functions (`LAG`, `DENSE_RANK`) to surface risk insights.
3. **`03_model.py`** — Feature engineering (transaction velocity, device
   switching, odd-hour activity, merchant risk profiling) and a Random Forest
   fraud classifier, with class imbalance handled via a manually implemented
   SMOTE (the `imbalanced-learn` package wasn't installable in this offline
   environment, so k-NN based synthetic minority oversampling is implemented
   from scratch using `scikit-learn`'s `NearestNeighbors`).
4. **`04_dashboard.py`** — Builds an executive-style dashboard summarizing
   the risk findings (`dashboard.png`).

## Key Findings (real output of this code)

- **Chargeback concentration:** The top 3 riskiest merchant categories
  (Crypto Exchange, Travel, Electronics) account for **86.1%** of total
  chargeback value, despite being 3 of 10 categories — a clear target for
  category-level risk controls.
- **Velocity risk:** Transactions occurring **<5 minutes** after a user's
  previous transaction show a **24.4% fraud rate**, versus a ~1.7% baseline
  fraud rate across all transactions — a strong, actionable velocity signal.
- **Model performance** (Random Forest + custom SMOTE, evaluated on a
  held-out test set with the original class imbalance, threshold tuned on a
  separate validation split): **F1-score 0.33, ROC-AUC 0.75, Recall 76%,
  Precision 21%.** The precision/recall trade-off is deliberate — in fraud
  detection, missing fraud (false negatives) is typically costlier than
  investigating a false alarm, so the threshold was tuned to prioritize
  recall.
- **Top predictive features:** merchant-category historical fraud rate,
  payment method, and odd-hour activity were the strongest signals.
- **Query performance:** aggregate risk queries over the 1.2M-row table
  averaged **~470ms** on SQLite with basic indexing (single-core sandboxed
  environment; would be materially faster on production infrastructure or
  with a columnar warehouse like Snowflake/BigQuery).

## Honest limitations

- Data is simulated, not real transactions.
- Trained on a 180K-row stratified subsample of the full 1.2M rows (all fraud
  cases + a random non-fraud sample) due to single-CPU compute constraints in
  this environment — features themselves were computed using each user's full
  transaction history first.
- `merchant_category_fraud_rate` is a target-encoded feature computed on the
  full dataset, which carries a mild risk of information leakage; in a
  production setting this would be computed strictly from training-fold data
  only.

## Stack

Python (Pandas, NumPy, scikit-learn), SQL (SQLite — CTEs, window functions),
Matplotlib (dashboard), Git.
