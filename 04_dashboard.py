import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cat_df = pd.read_csv(os.path.join(BASE_DIR, "output_category_risk.csv"))
trend_df = pd.read_csv(os.path.join(BASE_DIR, "output_success_trend.csv"))
gap_df = pd.read_csv(os.path.join(BASE_DIR, "output_velocity_risk.csv"))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Payment Risk & Fraud Analytics — Executive Dashboard (Simulated Data)", fontsize=14, fontweight="bold")

# 1. Chargeback amount by category
ax = axes[0, 0]
cat_sorted = cat_df.sort_values("chargeback_amount", ascending=True)
ax.barh(cat_sorted["merchant_category"], cat_sorted["chargeback_amount"], color="#c0392b")
ax.set_title("Chargeback Value by Merchant Category")
ax.set_xlabel("Chargeback amount ($)")

# 2. Success rate trend
ax = axes[0, 1]
ax.plot(trend_df["month"], trend_df["success_rate_pct"], marker="o", color="#2471a3")
ax.set_title("Monthly Payment Success Rate")
ax.set_ylabel("Success rate (%)")
ax.tick_params(axis="x", rotation=60)

# 3. Fraud rate by transaction gap bucket (velocity risk)
ax = axes[1, 0]
order = ["<5 min", "5-30 min", "30min-1day", ">1 day"]
gap_df["gap_bucket"] = pd.Categorical(gap_df["gap_bucket"], categories=order, ordered=True)
gap_df = gap_df.sort_values("gap_bucket")
ax.bar(gap_df["gap_bucket"], gap_df["fraud_rate_pct"], color="#e67e22")
ax.set_title("Fraud Rate by Time Since User's Last Transaction")
ax.set_ylabel("Fraud rate (%)")

# 4. Chargeback rate % by category
ax = axes[1, 1]
cat_sorted2 = cat_df.sort_values("chargeback_rate_pct", ascending=True)
ax.barh(cat_sorted2["merchant_category"], cat_sorted2["chargeback_rate_pct"], color="#8e44ad")
ax.set_title("Chargeback Rate % by Merchant Category")
ax.set_xlabel("Chargeback rate (%)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = os.path.join(BASE_DIR, "dashboard.png")
plt.savefig(output_path, dpi=140)
print(f"Saved {output_path}")