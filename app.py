import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Define the Streamlit app
st.title("Student Exam Performance Indicator")
st.write("Student Exam Performance Prediction")

# Create form for user input
with st.form("student_performance_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Race or Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", [
        "Associate's Degree", "Bachelor's Degree", "High School", "Master's Degree", "Some College", "Some High School"
    ])
    lunch = st.selectbox("Lunch Type", ["Free/Reduced", "Standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["None", "Completed"])
    writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100, step=1)
    reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100, step=1)
    
    submit_button = st.form_submit_button("Predict your Maths Score")

# Handle form submission
if submit_button:
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    pred_df = data.get_data_as_data_frame()
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    st.success(f"The predicted Maths score is: {results[0]}")

# Add custom CSS for additional styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stSelectbox select {
        background-color: #f9f9f9;
        border: none;
        color: #333;
        padding: 12px;
        font-size: 16px;
    }
    .stNumber_input input {
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
