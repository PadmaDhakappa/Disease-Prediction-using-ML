# Disease Prediction Using Machine Learning

This project implements a machine learning-based system to predict the likelihood of various diseases, including **Diabetes**, **Heart Disease**, and **Parkinson's Disease**, using a user-friendly interface built with **Streamlit**. The models are trained on publicly available datasets and aim to provide accurate predictions based on user input.

---

## Features

- **Diabetes Prediction:**
  - Predicts the likelihood of diabetes based on inputs like glucose levels, blood pressure, BMI, etc.
- **Heart Disease Prediction:**
  - Predicts the likelihood of heart disease based on features like cholesterol levels, heart rate, and other factors.
- **Parkinson's Disease Prediction:**
  - Detects the possibility of Parkinson's disease based on voice and signal processing features.
- Interactive and intuitive user interface using **Streamlit**.

---

## Technologies Used

- **Python**
- **Scikit-learn**: For machine learning models.
- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation.
- **Joblib**: For saving and loading machine learning models.

---

## Datasets

- **Diabetes Dataset:** A dataset containing medical records related to diabetes.
- **Heart Disease Dataset:** Features related to heart conditions.
- **Parkinson's Disease Dataset:** Voice signal features for Parkinson’s detection.

---

## Installation and Setup

Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/PadmaDhakappa/Disease-Prediction-using-ML.git


## Project Structure
Disease-Prediction-using-ML/
- **diabetes.csv**               - **(Diabetes dataset)**
- **heart.csv**                  - **Heart disease dataset**
- **parkinsons.csv**             - **Parkinson's disease dataset**
- **diabetes_model.pkl**         - **Trained diabetes prediction model**
- **heart_model.pkl**            - **Trained heart disease prediction model**
- **parkinsons_model.pkl**       - **Trained Parkinson's prediction model**
- **main.py**                    - **Streamlit app entry point**
- **requirements.txt**           - **Python dependencies**
- **README.md**                  - **Project documentation**


## Usage Instructions
1. Select a Prediction Type:
Use the sidebar to choose between Diabetes, Heart Disease, or Parkinson's Disease.

2. Enter Input Data:
Fill in the required fields for the selected prediction type.

3. Get the Prediction:
Click on the "Test Result" button to see the prediction outcome.
