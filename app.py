# ============================================================
# CUSTOMER CHURN PREDICTION — STREAMLIT WEB APP
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD SAVED MODEL, SCALER, COLUMNS
# ============================================================

# joblib.load = loads the saved file back into Python
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('columns.pkl')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ============================================================
# HEADER SECTION
# ============================================================

st.title("📡 Customer Churn Prediction")
st.markdown("Enter customer details below to predict whether they are likely to **churn (leave)** or **stay**.")
st.divider()

# ============================================================
# INPUT FORM — USER ENTERS CUSTOMER DETAILS
# ============================================================

st.subheader("👤 Customer Information")

# We split the form into 2 columns for cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.divider()
st.subheader("💳 Billing Information")

col3, col4 = st.columns(2)

with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col4:
    payment_method = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0)

st.divider()

# ============================================================
# PREDICTION LOGIC
# ============================================================

def predict_churn():
    # STEP 1 — Build a dictionary of all inputs
    # Keys must match EXACTLY what was in training data
    input_dict = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # STEP 2 — Convert to DataFrame (single row)
    input_df = pd.DataFrame([input_dict])

    # STEP 3 — One-hot encode exactly like training
    input_encoded = pd.get_dummies(input_df)

    # STEP 4 — Add any missing columns (with 0)
    # During training we had 29 feature columns
    # User input might be missing some dummy columns
    # e.g. if user selects "DSL", columns for "Fiber" won't exist
    # We add them back as 0
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # STEP 5 — Keep only the columns used during training, in same order
    input_encoded = input_encoded[feature_columns]

    # STEP 6 — Scale using the saved scaler
    input_scaled = scaler.transform(input_encoded)

    # STEP 7 — Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    # predict_proba gives probability: [prob of class 0, prob of class 1]

    return prediction, probability


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Churn", use_container_width=True):
    prediction, probability = predict_churn()

    st.divider()
    st.subheader("📊 Prediction Result")

    churn_probability = probability[1] * 100      # probability of churning
    stay_probability = probability[0] * 100        # probability of staying

    if prediction == 1:
        st.error("⚠️ This customer is **LIKELY TO CHURN**")
    else:
        st.success("✅ This customer is **LIKELY TO STAY**")

    # Show probability bars
    st.markdown("#### Confidence")
    st.metric("Churn Probability", f"{churn_probability:.1f}%")
    st.progress(int(churn_probability))

    st.metric("Stay Probability", f"{stay_probability:.1f}%")
    st.progress(int(stay_probability))

    # Extra advice based on result
    st.divider()
    st.subheader("💡 Recommendation")

    if prediction == 1:
        st.markdown("""
        **Actions to retain this customer:**
        - 🎁 Offer a loyalty discount or promotional deal
        - 📞 Schedule a customer service follow-up call
        - 🔒 Suggest upgrading to a 1 or 2 year contract
        - ⚡ Check if they have complaints or service issues
        """)
    else:
        st.markdown("""
        **This customer looks stable! You can:**
        - 🌟 Offer premium service upgrades
        - 📦 Suggest add-on services they don't have yet
        - 💌 Send a satisfaction survey to maintain engagement
        """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with Scikit-learn + Streamlit | Customer Churn Prediction Project</p>",
    unsafe_allow_html=True
)