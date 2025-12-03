import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_churn_model.pkl')
        return model
    except:
        st.error("Model file not found!")
        return None

model = load_model()

# App title
st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📊 Telco Customer Churn Prediction")
st.markdown("Predict whether a customer will churn or not")

# Sidebar for input
st.sidebar.header("Customer Information")

# Input fields
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox("Contract", 
    ["Month-to-month", "One year", "Two years"])
internet_service = st.sidebar.selectbox("Internet Service",
    ["DSL", "Fiber optic", "No"])

payment_method = st.sidebar.selectbox("Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", 
     "Credit card (automatic)"])

# Additional features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Default values for other features
multiple_lines = "No"
online_security = "No"
online_backup = "No"
device_protection = "No"
tech_support = "No"
streaming_tv = "No"
streaming_movies = "No"

# Prepare input data
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Prediction button
if st.sidebar.button("Predict Churn", type="primary"):
    if model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_df)[0]
                churn_prob = probability[1] if len(probability) > 1 else probability[0]
            else:
                churn_prob = 0.5 if prediction == 1 else 0.5
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.error(f"**CHURN DETECTED**")
                    st.write(f"Churn Probability: {churn_prob:.2%}")
                else:
                    st.success(f"**NO CHURN**")
                    st.write(f"Churn Probability: {churn_prob:.2%}")
            
            with col2:
                # Gauge chart
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.barh([0], [churn_prob], color='red' if prediction == 1 else 'green')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Churn Probability')
                ax.set_title('Churn Risk')
                st.pyplot(fig)
            
            # Recommendations
            st.subheader("Recommendations")
            if prediction == 1:
                st.warning("""
                **High Churn Risk Detected!**
                - Offer retention discount (10-15% for 3 months)
                - Provide personalized service review
                - Schedule follow-up call within 48 hours
                - Consider service upgrade offer
                """)
            else:
                st.info("""
                **Customer is Stable**
                - Maintain current service quality
                - Consider upselling opportunities
                - Regular satisfaction check every 6 months
                - Invite to loyalty program
                """)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    else:
        st.error("Model not loaded. Please check if model file exists.")

# Model information
with st.expander("Model Information"):
    st.write("""
    **Model Details:**
    - Algorithm: Ensemble Model (Voting Classifier)
    - Best F1-Score: 0.62
    - Accuracy: 0.81
    - Precision: 0.67
    - Recall: 0.58
    
    **Training Data:**
    - Telco Customer Churn Dataset
    - 7,043 customer records
    - 20 features
    
    **Features Used:**
    - Contract type (most important)
    - Internet service type
    - Payment method
    - Tenure (duration)
    - Monthly charges
    - Total charges
    
    **Note:** This is a demo application for academic purposes.
    Bengkel Koding Data Science - UDINUS 2025/2026
    """)

# Footer
st.markdown("---")
st.markdown("**Bengkel Koding Data Science** • Universitas Dian Nuswantoro • 2025/2026")
st.markdown("*For academic purposes only*")
