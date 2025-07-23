import streamlit as st
import numpy as np
import pickle

# Load models
logistic_model = pickle.load(open("L_model", "rb"))
svm_model = pickle.load(open("svm_model", "rb"))
dt_model = pickle.load(open("DT_model", "rb"))
rf_model = pickle.load(open("RF_model", "rb"))

# Page setup
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# Custom CSS using only specified colors
st.markdown("""
    <style>
    .stApp {
        background-color: #273F4F;
        color: #000000;
    }

    h1 {
        color: #FF7A30;
        text-align: center;
        font-size: 42px;
        padding-bottom: 10px;
    }

    .block-container {
        padding: 2rem 2rem;
    }

    .stButton>button {
        background-color: #FF7A30;
        color: #000000;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #465C88;
        color: white;
    }

    .stSidebar, .css-1d391kg, .css-1d3w5wq {
        background-color: #465C88 !important;
        color: #E9E3DF;
    }

    .st-radio label, .st-selectbox label, .st-slider label, .st-number-input label {
        color: #000000 !important;
    }

    .st-radio div, .st-selectbox div, .st-slider div, .st-number-input div {
        background-color: white !important;
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1> 🎓 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Enter the student's academic and lifestyle details below to predict performance.")

# Sidebar model selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio("Select Model:", (
    "Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest"))

st.sidebar.markdown("---")
st.sidebar.write("📚 Models are trained on real student data.")

# Input fields
st.markdown("---")
st.markdown("### 📝 Student Information")

age = st.slider("Age", 10, 25, 18)
study_hours = st.slider("Daily Study Hours", 0, 12, 3)
gpa = st.slider("📘 GPA (0.0 - 4.0)", 0.0, 4.0, 3.0, step=0.1)
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Number of Absences", 0, 50, 5)
school_support = st.radio("School Support", ["Yes", "No"])
internet = st.radio("Internet Access", ["Yes", "No"])
parent_edu = st.slider("Parental Education Level (1 to 5)", 1, 5, 3)

# Encode categorical inputs
school_support = 1 if school_support == "Yes" else 0
internet = 1 if internet == "Yes" else 0

# Prepare input array
input_data = np.array([[age, study_hours, failures, absences, school_support, internet, parent_edu, gpa]])

# Define grade categories
grades = {
    0: "Fail",
    1: "Pass",
    2: "Good",
    3: "Very Good",
    4: "Excellent"
}

# Prediction
if st.button("📊 Predict Performance"):
    if model_option == "Logistic Regression":
        prediction = logistic_model.predict(input_data)
    elif model_option == "Support Vector Machine":
        prediction = svm_model.predict(input_data)
    elif model_option == "Decision Tree":
        prediction = dt_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    result = grades.get(prediction[0], "Unknown")
    st.markdown("---")
    st.success(f"🎯 Predicted Grade: **{result}**")
