# app.py

import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load data and model
df = pd.read_csv("cleaned_fintech_data.csv")
model = joblib.load("bankruptcy_rf_model.pkl")

# Title
st.title("🚀 Fintech Bankruptcy Prediction App")

# User input form
st.subheader("Enter Financial Metrics for Bankruptcy Prediction")

sample_input = df.drop(columns=['Bankrupt?', 'Cluster']).sample(1).copy()

user_input = {}
for col in sample_input.columns:
    user_input[col] = st.number_input(f"{col}", value=float(sample_input[col].values[0]))

if st.button("Predict Bankruptcy"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success("Prediction: **Bankrupt**" if prediction == 1 else "Prediction: **Not Bankrupt**")

# Show dataset preview
with st.expander("📊 View Cleaned Dataset"):
    st.dataframe(df.head())

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Saurabh | Fintech AI Strategy")
