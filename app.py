import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import joblib

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the model and encoders
@st.cache_resource
def load_models():
    model = load_model('model1.h5')
    scaler = joblib.load('scaler.pkl')
    onehot_encoder = joblib.load('onehotencoder.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, onehot_encoder, label_encoder

# Load the models
try:
    model, scaler, onehot_encoder, label_encoder = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Title and description
st.title("Customer Churn Prediction")
st.markdown("""
This application predicts whether a customer will churn or not based on their characteristics.
Please fill in the customer details below to make a prediction.
""")

# Create input fields
st.header("Customer Information")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    # Numerical features
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

with col2:
    # Categorical features
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])

# Create a button for prediction
if st.button("Predict Churn"):
    try:
        # Prepare the input data
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale numerical features
        numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Encode categorical features
        categorical_features = ['Geography', 'Gender']
        encoded_cats = onehot_encoder.transform(input_df[categorical_features])
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=onehot_encoder.get_feature_names_out(categorical_features))
        
        # Combine numerical and categorical features
        final_input = pd.concat([input_df.drop(categorical_features, axis=1), encoded_cats_df], axis=1)
        
        # Make prediction
        prediction = model.predict(final_input)
        churn_probability = prediction[0][0]
        
        # Display results
        st.header("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Churn Probability", f"{churn_probability:.2%}")
        
        with col2:
            if churn_probability > 0.5:
                st.error("Customer is likely to churn")
            else:
                st.success("Customer is likely to stay")
                
        # Display feature importance or additional insights
        st.subheader("Customer Profile")
        st.write("""
        Based on the input data, here are some key insights:
        - Credit Score: {credit_score}
        - Age: {age}
        - Tenure: {tenure} years
        - Geography: {geography}
        - Number of Products: {num_of_products}
        - Active Member: {is_active_member}
        """.format(**input_data))
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by TensorFlow") 