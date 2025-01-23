import streamlit as st
import numpy as np
import joblib

# Load models
diabetes_model = joblib.load('diabetes_model.pkl')
heart_model = joblib.load('heart_model.pkl')
parkinsons_model = joblib.load('parkinsons_model.pkl')

# Sidebar options
st.sidebar.title("Prediction of Disease Outbreaks System")
option = st.sidebar.radio(
    "Choose a prediction type:",
    ("Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction")
)

# Diabetes Prediction
if option == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300)
    bp = st.number_input("Blood Pressure value", min_value=0, max_value=200)
    skin_thickness = st.number_input("Skin Thickness value", min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900)
    bmi = st.number_input("BMI value", min_value=0.0, max_value=100.0)
    dpf = st.number_input("Diabetes Pedigree Function value", min_value=0.0, max_value=10.0)
    age = st.number_input("Age of the Person", min_value=0, max_value=120, step=1)
    
    if st.button("Diabetes Test Result"):
        input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)[0]
        if prediction == 1:
            st.error("The person is likely to have diabetes.")
        else:
            st.success("The person is unlikely to have diabetes.")

# Heart Disease Prediction
elif option == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300)
    chol = st.number_input("Cholesterol Level", min_value=0, max_value=600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = st.number_input("Resting Electrocardiographic Results (0-2)", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0)
    slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1)
    thal = st.number_input("Thalassemia (0-3)", min_value=0, max_value=3, step=1)
    
    if st.button("Heart Disease Test Result"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)[0]
        if prediction == 1:
            st.error("The person is likely to have heart disease.")
        else:
            st.success("The person is unlikely to have heart disease.")

# Parkinson's Prediction
elif option == "Parkinson's Prediction":
    st.title("Parkinson's Prediction using ML")
    
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=400.0)
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0)
    jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0)
    jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.1)
    rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=1.0)
    ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=1.0)
    ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=1.0)
    shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0)
    shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=10.0)
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0)
    hnr = st.number_input("HNR", min_value=0.0, max_value=50.0)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0)
    dfa = st.number_input("DFA", min_value=0.0, max_value=1.0)
    spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0)
    spread2 = st.number_input("spread2", min_value=0.0, max_value=1.0)
    d2 = st.number_input("D2", min_value=0.0, max_value=5.0)
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0)
    
    if st.button("Parkinson's Test Result"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(input_data)[0]
        if prediction == 1:
            st.error("The person is likely to have Parkinson's disease.")
        else:
            st.success("The person is unlikely to have Parkinson's disease.")
