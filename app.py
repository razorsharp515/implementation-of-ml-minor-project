from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from fraud_detection.config import MODEL_PATH, TRANSACTION_TYPES


st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("Financial Transaction Fraud Detection")

if not MODEL_PATH.exists():
    st.warning("Train the model first with: python -m fraud_detection.train")
    st.stop()

bundle = joblib.load(MODEL_PATH)
feature_names = bundle["feature_names"]

with st.form("transaction"):
    step = st.number_input("Step / hour", min_value=1, max_value=744, value=10)
    transaction_type = st.selectbox("Transaction type", TRANSACTION_TYPES, index=1)
    amount = st.number_input("Amount", min_value=0.0, value=25000.0, step=1000.0)
    oldbalance_org = st.number_input("Origin old balance", min_value=0.0, value=30000.0, step=1000.0)
    oldbalance_dest = st.number_input("Destination old balance", min_value=0.0, value=5000.0, step=1000.0)
    submitted = st.form_submit_button("Check transaction")

if submitted:
    row = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
    }
    for name in feature_names:
        if name.startswith("type_"):
            row[name] = 1 if name == f"type_{transaction_type}" else 0

    frame = pd.DataFrame([row])
    for feature in feature_names:
        if feature not in frame.columns:
            frame[feature] = 0
    frame = frame[feature_names]
    scaled = pd.DataFrame(bundle["scaler"].transform(frame), columns=feature_names)
    probability = float(bundle["model"].predict_proba(scaled)[0, 1])

    st.metric("Fraud probability", f"{probability:.2%}")
    if probability >= 0.5:
        st.error("Prediction: Fraud")
    else:
        st.success("Prediction: Not fraud")
