# app.py
import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("income_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Employee Income Predictor", layout="centered")
st.title("💼 Employee Income Prediction")

# 🚀 Collect user inputs
age = st.slider("Age", 18, 90, 30)
education = st.selectbox("Education", encoders['education'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
gender = st.selectbox("Gender", encoders['gender'].classes_)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)

if st.button("Predict Income"):
    # ✨ Prepare input
    input_data = {
        "age": age,
        "education": encoders["education"].transform([education])[0],
        "occupation": encoders["occupation"].transform([occupation])[0],
        "gender": encoders["gender"].transform([gender])[0],
        "hours-per-week": hours_per_week
    }

    # Add missing features with default value (e.g., 0)
    for col in model.feature_names_in_:
        if col not in input_data:
            input_data[col] = 0

    input_vector = np.array([input_data[col] for col in model.feature_names_in_]).reshape(1, -1)
    st.write("🔍 Transformed Input Vector:", input_vector)

    # 🔮 Predict
    prediction = model.predict(input_vector)
    prediction_label = encoders["income"].inverse_transform(prediction)[0]

    # 🎉 Output
    
    st.success(f"💸 Predicted Income: {prediction_label}")
