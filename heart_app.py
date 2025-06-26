import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("heart_model.joblib")  # Make sure this file exists in the same directory

# Load data for feature reference
data = pd.read_csv("heart.csv")

# App title
st.title("Heart Disease Prediction App")
st.markdown("Enter patient data to predict the likelihood of heart disease.")

# Sidebar input
st.sidebar.header("Patient Information")

def user_input_features():
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', sorted(data.cp.unique()))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', sorted(data.restecg.unique()))
    thalach = st.sidebar.slider('Max Heart Rate Achieved (thalach)', int(data.thalach.min()), int(data.thalach.max()), 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.slider('Oldpeak', float(data.oldpeak.min()), float(data.oldpeak.max()), 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', sorted(data.slope.unique()))
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy (ca)', sorted(data.ca.unique()))
    thal = st.sidebar.selectbox('Thalassemia (thal)', sorted(data.thal.unique()))

    # Encode sex
    sex_encoded = 1 if sex == 'Male' else 0

    features = {
        'sex': sex_encoded,
        'cp': cp,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame([features])

# Predict
input_df = user_input_features()

st.subheader("Patient Data")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
st.write("**Heart Disease Detected!**" if prediction[0] == 1 else "**No Heart Disease Detected.**")

st.subheader("Prediction Probability")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
