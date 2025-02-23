import numpy as np
import joblib
import streamlit as st

# Load model and encoders
model = joblib.load("model.pkl")  # Ensure this file exists
encoders = joblib.load("encoders.pkl")  # Load encoding dictionary

st.title("Customer Churn Prediction")

st.divider()
st.write("Please enter the following details to predict Customer Churn")
st.divider()

# User Inputs
Age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
Gender = st.selectbox("Enter Gender", ["Female", "Male"])
Tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=30)
Monthly_charges = st.number_input("Enter Monthly Charges", min_value=30, max_value=150)

st.divider()

# Encode categorical variables
def encode_feature(feature_name, user_input):
    """Encodes categorical input using saved encoders."""
    if feature_name in encoders:
        return encoders[feature_name].transform([user_input])[0]
    else:
        st.error(f"Error: {feature_name} column not found in encoders. Check preprocessing.")
        st.stop()  # Stop execution if encoding key is missing

gender_selected = encode_feature("Gender", Gender)  # Encode Gender correctly

predict_button = st.button("Predict!")

if predict_button:
    # Ensure input matches training format
    x = np.array([[Age, gender_selected, Tenure, Monthly_charges]])

    # Make prediction (No scaling applied)
    prediction = model.predict(x)[0]

    # Convert output to readable format
    predicted = "Churn" if prediction == 'Yes' else "Not Churn"
    
    st.write(f"The Customer is predicted to {predicted}")

else:
    st.write("Please enter details and click Predict.")
