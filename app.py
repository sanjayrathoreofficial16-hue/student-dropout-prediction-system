import pandas as pd
import streamlit as st
import joblib

st.title("Student Dropout Prediction System")

# Load dataset for input ranges
df = pd.read_csv("data.csv", sep=';')

# Clean column names
df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\t', '')

# Load model, scaler, and label encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Display dataset
st.subheader("Dataset Preview")
st.write(df.head())

# User Input
st.subheader("Enter Student Details")

# Dictionary to store user inputs (only features, exclude Target)
user_input = {}
feature_columns = df.columns[:-1]  # Exclude 'Target'

for col in feature_columns:
    # All features are numeric (int64/float64), so use number_input
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=min_val)

# Predict button
if st.button("Predict Dropout"):
    # Create DataFrame from inputs (ensure column order matches training)
    input_df = pd.DataFrame([user_input])[feature_columns]
    
    # Scale input
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred = model.predict(input_scaled)
    pred_label = le.inverse_transform(pred)[0]
    
    st.write("Entered Student Details:")
    st.write(user_input)
    st.success(f"Prediction: {pred_label}")