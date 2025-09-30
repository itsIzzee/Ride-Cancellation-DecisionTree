import streamlit as st
import joblib
import pandas as pd

# --- Page setup (must be the first Streamlit command) ---
st.set_page_config(page_title="Ride Cancellation Predictor", layout="centered")

# --- Load the trained model ---
@st.cache_resource
def load_model():
    return joblib.load("decision_tree.pkl")

model = load_model()

# --- App Title ---
st.title("🚖 Ride Cancellation Prediction")
st.markdown(
    "Use this tool to predict whether a ride booking will be **successful or cancelled** "
    "based on booking details."
)

st.divider()

# --- Define options ---
areas = [f"Area-{i}" for i in range(1, 51)]
days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
vehicles = ["Auto", "Mini", "Sedan", "Bike"]

# --- Input form ---
st.subheader("📋 Booking Details")

col1, col2 = st.columns(2)
with col1:
    pickup_location = st.selectbox("Pickup Location", areas)
    vehicle_type = st.selectbox("Vehicle Type", vehicles)
    customer_rating = st.number_input("Customer Rating (1–5)", 1.0, 5.0, 4.5)
with col2:
    drop_location = st.selectbox("Drop Location", areas)
    payment_method = st.selectbox("Payment Method", ["Cash", "Card"])
    hour_of_day = st.slider("Hour of Day", 0, 23, 12)

col3, col4 = st.columns(2)
with col3:
    day_of_week = st.selectbox("Day of Week", days)
with col4:
    is_weekend = 1 if day_of_week in ["Saturday","Sunday"] else 0
    st.write("Weekend?", "✅ Yes" if is_weekend else "❌ No")

st.divider()

# --- Derived features ---
def assign_time_band(h):
    if 5 <= h <= 11:
        return "Morning"
    elif 12 <= h <= 16:
        return "Afternoon"
    elif 17 <= h <= 21:
        return "Evening"
    else:
        return "Night"

time_band = assign_time_band(hour_of_day)
pickup_cancel_rate = 0.0
drop_cancel_rate = 0.0
pickup_drop_pair_freq = 0

# --- Build DataFrame with required training columns ---
input_df = pd.DataFrame([{
    "hour_of_day": hour_of_day,
    "day_of_week": days.index(day_of_week),
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

# --- Prediction section ---
st.subheader("🔮 Prediction")
if st.button("Predict Ride Status", use_container_width=True):
    prediction = model.predict(input_df)[0]

    if prediction.lower() == "success":
        st.success(f"✅ Predicted Booking Status: **{prediction}**")
    elif "cancel" in prediction.lower():
        st.error(f"❌ Predicted Booking Status: **{prediction}**")
    else:
        st.warning(f"⚠️ Predicted Booking Status: **{prediction}**")
