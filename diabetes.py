import streamlit as st
import numpy as np
import joblib  # To load the trained model

# Load the trained model
model = joblib.load('diabetes_model.pkl')  # Ensure 'diabetes_model.pkl' exists in the same directory

# Streamlit UI
st.title("Diabetes Prediction App")

st.sidebar.header("Input Parameters")
# Input fields for user data
pregnancies = st.sidebar.number_input("Number of Pregnancies", 0, 20, step=1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure Value", 0, 150, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness Value", 0, 100, step=1)
insulin = st.sidebar.number_input("Insulin Level", 0, 900, step=1)
bmi = st.sidebar.number_input("BMI Value", 0.0, 100.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function Value", 0.0, 2.5, step=0.01)
age = st.sidebar.number_input("Age of the Person", 0, 120, step=1)

# Collect input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# Predict diabetes
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("The model predicts that the person is diabetic.")
    else:
        st.success("The model predicts that the person is not diabetic.")
