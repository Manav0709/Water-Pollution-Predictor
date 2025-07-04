import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and column structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# List of pollutants
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# --- Sidebar Inputs ---
st.sidebar.header("Enter Input Details")
year_input = st.sidebar.number_input("Select Year", min_value=2000, max_value=2100, value=2022, step=1)
station_id = st.sidebar.text_input("Enter Station ID", value='1')

# --- Main Title ---
st.title("💧 Water Pollution Predictor")
st.markdown("""
This app predicts the **concentration of 6 key water pollutants** 
based on the selected **Year** and **Station ID**.
""")

st.markdown("---")

# --- Predict Button ---
if st.sidebar.button("🚀 Predict"):
    # Input preparation
    input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
    input_encoded = pd.get_dummies(input_df, columns=['id'])

    # Align with model columns
    for col in model_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_cols]

    # Make prediction
    predicted = model.predict(input_encoded)[0]

    # --- Display Output ---
    st.subheader(f"📊 Predicted pollutant levels for station **{station_id}** in **{year_input}**:")
    result_df = pd.DataFrame({
        "Pollutant": pollutants,
        "Predicted Level": [f"{val:.2f}" for val in predicted]
    })

    st.table(result_df)

else:
    st.info("🔍 Enter values in the sidebar and click **Predict** to see results.")

# Optional footer
st.markdown("""
<hr style='border:1px solid lightgray'>
<small>Made with ❤️ using Streamlit</small>
""", unsafe_allow_html=True)
