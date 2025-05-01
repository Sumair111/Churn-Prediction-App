import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load('churn_prediction.pkl')

# Feature names (must match training order)
feature_names = [
    'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay',
    'Total Spend', 'Last Interaction', 'Gender_Male',
    'Subscription Type_Premium', 'Subscription Type_Standard',
    'Contract Length_Monthly', 'Contract Length_Quarterly'
]

# Try to get feature importance if model supports it
try:
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
except:
    feature_importance = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["🔮 Churn Prediction", "📊 Feature Importance"])

# =======================
# Page 1: Churn Prediction
# =======================
if page == "🔮 Churn Prediction":
    st.title('Churn Prediction Web App')

    st.header("📋 Enter Customer Details")
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    tenure = st.number_input('Tenure (months)', min_value=1, max_value=100, value=24)
    usage_frequency = st.number_input('Usage Frequency', min_value=1, max_value=50, value=10)
    support_calls = st.number_input('Support Calls', min_value=0, max_value=20, value=2)
    payment_delay = st.number_input('Payment Delay (days)', min_value=0, max_value=50, value=5)
    total_spend = st.number_input('Total Spend', min_value=50, max_value=1000, value=200)
    last_interaction = st.number_input('Last Interaction (days)', min_value=0, max_value=30, value=10)

    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_male = 1 if gender == "Male" else 0

    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    sub_premium = 1 if subscription == "Premium" else 0
    sub_standard = 1 if subscription == "Standard" else 0

    contract = st.selectbox("Contract Length", ["Annual", "Quarterly", "Monthly"])
    contract_monthly = 1 if contract == "Monthly" else 0
    contract_quarterly = 1 if contract == "Quarterly" else 0

    # DataFrame
    user_input = pd.DataFrame({
        'Age': [age],
        'Tenure': [tenure],
        'Usage Frequency': [usage_frequency],
        'Support Calls': [support_calls],
        'Payment Delay': [payment_delay],
        'Total Spend': [total_spend],
        'Last Interaction': [last_interaction],
        'Gender_Male': [gender_male],
        'Subscription Type_Premium': [sub_premium],
        'Subscription Type_Standard': [sub_standard],
        'Contract Length_Monthly': [contract_monthly],
        'Contract Length_Quarterly': [contract_quarterly]
    })

    # Prediction
    if st.button("🔍 Predict"):
        prediction = model.predict(user_input)
        if prediction[0] == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")

# ==============================
# Page 2: Feature Importance
# ==============================
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance Visualization")
    if feature_importance is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance in Churn Prediction', fontsize=16)
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        st.pyplot(plt.gcf())
    else:
        st.warning("Feature importance not available for this model.")
