import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model_data = joblib.load("best_bean_model.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]

# Streamlit page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="centered"
)

st.title("🫘 Dry Bean Classification")
st.write("Enter the physical measurements of a dry bean to predict its class.")

st.subheader("Enter Bean Measurements")

with st.form("bean_form"):

    eccentricity = st.number_input("Eccentricity", min_value=0.0, value=0.0)
    convex_area = st.number_input("Convex Area", min_value=0.0, value=0.0)
    extent = st.number_input("Extent", min_value=0.0, value=0.0)
    solidity = st.number_input("Solidity", min_value=0.0, value=0.0)
    roundness = st.number_input("Roundness", min_value=0.0, value=0.0)
    shape_factor1 = st.number_input("Shape Factor 1", min_value=0.0, value=0.0)
    shape_factor4 = st.number_input("Shape Factor 4", min_value=0.0, value=0.0)

    submit = st.form_submit_button("Predict Bean Class")

    if submit:

        # Collect user inputs
        input_data = {
            "Eccentricity": eccentricity,
            "ConvexArea": convex_area,
            "Extent": extent,
            "Solidity": solidity,
            "roundness": roundness,
            "ShapeFactor1": shape_factor1,
            "ShapeFactor4": shape_factor4
        }

        # Convert inputs into DataFrame in model feature order
        input_df = pd.DataFrame([input_data])[features]

        # Apply the same scaling used during training
        input_scaled = scaler.transform(input_df)

        # Predict bean class
        prediction = model.predict(input_scaled)

        # Display result
        st.success(f"Predicted Bean Class: {prediction[0]}")