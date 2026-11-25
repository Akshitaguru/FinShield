import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Payment Risk & Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load generated data
cat_df = pd.read_csv(os.path.join(BASE_DIR, "output_category_risk.csv"))
trend_df = pd.read_csv(os.path.join(BASE_DIR, "output_success_trend.csv"))
gap_df = pd.read_csv(os.path.join(BASE_DIR, "output_velocity_risk.csv"))

# --- Header ---
st.title("🛡️ Real-Time Payment Risk & Fraud Analytics Pipeline")
st.caption("Live monitoring of transaction velocity, merchant category risk, and chargeback exposure.")

# --- Top KPI Summary Cards ---
col1, col2, col3, col4 = st.columns(4)

total_txns = trend_df["total_txns"].sum()
total_chargeback = cat_df["chargeback_amount"].sum()
avg_success_rate = trend_df["success_rate_pct"].mean()
high_velocity_fraud = gap_df.loc[gap_df["gap_bucket"] == "<5 min", "fraud_rate_pct"].values[0]

with col1:
    st.metric("Total Transactions Monitored", f"{total_txns:,}")
with col2:
    st.metric("Total Chargeback Exposure", f"${total_chargeback:,.2f}")
with col3:
    st.metric("Avg Success Rate", f"{avg_success_rate:.2f}%")
with col4:
    st.metric("High-Velocity Fraud Rate (<5m)", f"{high_velocity_fraud:.2f}%")

st.markdown("---")

# --- Charts Row 1 ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Chargeback Value by Merchant Category")
    cat_sorted = cat_df.sort_values("chargeback_amount", ascending=True)
    fig_cat = px.bar(
        cat_sorted,
        x="chargeback_amount",
        y="merchant_category",
        orientation="h",
        color="chargeback_amount",
        color_continuous_scale="Reds",
        labels={"chargeback_amount": "Total Amount ($)", "merchant_category": "Category"},
    )
    fig_cat.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_cat, use_container_width=True)

with row1_col2:
    st.subheader("Monthly Gateway Success Rate Trend")
    fig_trend = px.line(
        trend_df,
        x="month",
        y="success_rate_pct",
        markers=True,
        labels={"month": "Month", "success_rate_pct": "Success Rate (%)"},
    )
    fig_trend.update_traces(line_color="#2471a3", line_width=3)
    fig_trend.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_trend, use_container_width=True)

# --- Charts Row 2 ---
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Velocity Risk: Fraud by Time Since Last Txn")
    order = ["<5 min", "5-30 min", "30min-1day", ">1 day"]
    gap_df["gap_bucket"] = pd.Categorical(gap_df["gap_bucket"], categories=order, ordered=True)
    gap_df_sorted = gap_df.sort_values("gap_bucket")

    fig_gap = px.bar(
        gap_df_sorted,
        x="gap_bucket",
        y="fraud_rate_pct",
        color="fraud_rate_pct",
        color_continuous_scale="Oranges",
        labels={"gap_bucket": "Time Window", "fraud_rate_pct": "Fraud Rate (%)"},
    )
    fig_gap.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_gap, use_container_width=True)

with row2_col2:
    st.subheader("Category Chargeback Rate (%)")
    cat_rate_sorted = cat_df.sort_values("chargeback_rate_pct", ascending=True)
    fig_rate = px.bar(
        cat_rate_sorted,
        x="chargeback_rate_pct",
        y="merchant_category",
        orientation="h",
        color="chargeback_rate_pct",
        color_continuous_scale="Purples",
        labels={"chargeback_rate_pct": "Chargeback Rate (%)", "merchant_category": "Category"},
    )
    fig_rate.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_rate, use_container_width=True)

# --- Live Model Simulation Feature ---
st.markdown("---")
st.subheader("🔍 Real-Time Transaction Risk Tester")
st.write("Input simulated transaction features to see immediate risk flags based on our pipeline rules.")

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    cat_input = st.selectbox("Merchant Category", cat_df["merchant_category"].unique())
    amount_input = st.number_input("Amount ($)", value=250.0, step=10.0)
with sim_col2:
    gap_input = st.slider("Minutes Since Last Transaction", 0, 120, 2)
    device_switch = st.checkbox("New Device Used?", value=True)
with sim_col3:
    hour_input = st.slider("Hour of Day (24h)", 0, 23, 2)

# Quick rule-based risk evaluation
risk_score = 0
reasons = []

if cat_input in ["Crypto Exchange", "Travel", "Electronics"]:
    risk_score += 35
    reasons.append(f"High-risk category ({cat_input})")
if gap_input < 5:
    risk_score += 30
    reasons.append("High velocity (<5 min gap)")
if device_switch and gap_input < 10:
    risk_score += 25
    reasons.append("Rapid device switch")
if hour_input <= 5:
    risk_score += 15
    reasons.append("Odd-hour transaction (12am-5am)")

if st.button("Evaluate Transaction Risk"):
    if risk_score >= 60:
        st.error(f"🚨 **High Risk Flagged (Score: {risk_score}/100)**: Decline / Route to manual review.")
    elif risk_score >= 30:
        st.warning(f"⚠️ **Moderate Risk (Score: {risk_score}/100)**: Require 2FA OTP verification.")
    else:
        st.success(f"✅ **Low Risk (Score: {risk_score}/100)**: Auto-approve.")
    
    if reasons:
        st.write("**Risk Factors Detected:** " + ", ".join(reasons))