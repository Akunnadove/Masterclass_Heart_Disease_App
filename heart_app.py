import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("heart_model.joblib")  # Ensure this file exists

# Load data for valid ranges
data = pd.read_csv("heart.csv")

# App title
st.title("Heart Disease Prediction App")
st.markdown("Provide the patient details below to check for risk of heart disease.")

# Input fields
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", sorted(data.cp.unique()))
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=90, max_value=200, value=120)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", sorted(data.restecg.unique()))
thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=70, max_value=210, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
ca = st.selectbox("Number of Major Vessels (ca)", sorted(data.ca.unique()))
thal = st.selectbox("Thalassemia (thal)", sorted(data.thal.unique()))

# Encode categorical variables
sex_encoded = 1 if sex == "Male" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    "sex": sex_encoded,
    "cp": cp,
    "trestbps": trestbps,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "ca": ca,
    "thal": thal
}])

# Show data
st.subheader("Patient Input Data")
st.write(input_data)

# Prediction
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result")
    st.success("💔 Heart Disease Detected!" if prediction[0] == 1 else "❤️ No Heart Disease Detected.")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Heart Disease: **{prediction_proba[0][1]:.2f}**")
