import streamlit as st
import joblib
import numpy as np

# Load your trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Water Potability Prediction App")
st.markdown("This application predicts whether water is potable based on input features.")

# Function to make predictions
def predict_potability(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

# Inputs from user
st.header("Enter Water Parameters:")

ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.slider("Hardness", min_value=0, max_value=500, value=100)
solids = st.slider("Total Dissolved Solids", min_value=0, max_value=2000, value=500)
chloramines = st.slider("Chloramines", min_value=0, max_value=10, value=2)
sulfate = st.slider("Sulfate", min_value=0, max_value=1000, value=200)
conductivity = st.slider("Conductivity", min_value=0, max_value=10000, value=2000)
organic_carbon = st.slider("Organic Carbon", min_value=0, max_value=50, value=10)
trihalomethanes = st.slider("Trihalomethanes", min_value=0, max_value=100, value=20)
turbidity = st.slider("Turbidity", min_value=0, max_value=10, value=1)

features = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]

if st.button("Predict Potability"):
    prediction = predict_potability(features)
    if prediction == 1:
        st.success("The water is **potable**!")
    else:
        st.error("The water is **not potable**.")
