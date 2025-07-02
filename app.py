import streamlit as st
import pandas as pd
import joblib

# Load the model and preprocessor
model = joblib.load("RF_Classifier_model.joblib")
preprocessor = joblib.load("preprocessor_file.joblib")

st.title("🧳 Travel Product Purchase Prediction App")

st.markdown("Fill in the details below to predict whether the customer will purchase the product.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (in minutes)", min_value=0, value=15)
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Others'])
gender = st.selectbox("Gender", ['Male', 'Female'])
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, value=2)
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
number_of_trips = st.number_input("Number of Trips", min_value=0, value=2)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("Monthly Income", min_value=0, value=30000)
total_visiting = st.number_input("Total People Visiting (adults + children)", min_value=0, value=3)

# Create DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [typeof_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfFollowups': [number_of_followups],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income],
    'TotalVisiting': [total_visiting]
})

# Predict
if st.button("Predict"):
    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)[0]
    
    if prediction == 1:
        st.success("✅ The customer is likely to **buy the product**.")
    else:
        st.warning("❌ The customer is **not likely to buy the product**.")
