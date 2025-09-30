import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("decision_tree.pkl")

model = load_model()

st.title("🚖 Ride Cancellation Prediction (Decision Tree)")

st.markdown("Enter ride booking details below to predict if the ride will be **successful or cancelled**.")

# Collect user inputs
pickup_location = st.text_input("Pickup Location (e.g., Area-1)")
drop_location = st.text_input("Drop Location (e.g., Area-2)")
vehicle_type = st.selectbox("Vehicle Type", ["Auto", "Mini", "Sedan", "Bike"])
payment_method = st.selectbox("Payment Method", ["Cash", "Card"])
customer_rating = st.number_input("Customer Rating (1–5)", 1.0, 5.0, 4.5)
hour_of_day = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
is_weekend = 1 if day_of_week in ["Saturday","Sunday"] else 0

# Derived / engineered fields with safe defaults
pickup_cancel_rate = 0.0
drop_cancel_rate = 0.0
pickup_drop_pair_freq = 0
time_band = None   # could also derive from hour_of_day if you want

# Build DataFrame with EXACT columns
input_df = pd.DataFrame([{
    "hour_of_day": hour_of_day,
    "day_of_week": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week),
    "is_weekend": is_weekend,
    "pickup_cancel_rate": pickup_cancel_rate,
    "drop_cancel_rate": drop_cancel_rate,
    "pickup_drop_pair_freq": pickup_drop_pair_freq,
    "customer_rating": customer_rating,
    "time_band": time_band,
    "Pickup Location": pickup_location,
    "Drop Location": drop_location,
    "vehicle_type": vehicle_type,
    "payment_method": payment_method
}])

# Predict
if st.button("Predict Ride Status"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Booking Status: **{prediction}**")
