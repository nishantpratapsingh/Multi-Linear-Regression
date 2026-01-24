import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Predict car prices using **Multiple Linear Regression**")

# Load model safely
try:
    with open("model.pkl", "rb") as file:
        model, preprocessor = pickle.load(file)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model could not be loaded")
    st.exception(e)
    st.stop()

# User inputs
brand = st.selectbox(
    "Select Car Brand",
    ["Maruti", "Hyundai", "Honda", "Toyota", "BMW"]
)

mileage = st.slider("Mileage (km/l)", 10, 30, 18)
engine_size = st.slider("Engine Size (cc)", 800, 3000, 1500)

# Prediction
if st.button("Predict Price"):
    try:
        input_df = pd.DataFrame({
            "Brand": [brand],
            "Mileage": [mileage],
            "Engine_Size": [engine_size]
        })

        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)

        st.success(f"💰 Estimated Car Price: ₹ {int(prediction[0]):,}")

        # Feature importance
        feature_names = (
            preprocessor.named_transformers_["brand"]
            .get_feature_names_out(["Brand"])
            .tolist()
            + ["Mileage", "Engine_Size"]
        )

        coeffs = model.coef_

        fig, ax = plt.subplots()
        ax.barh(feature_names, coeffs)
        ax.set_title("Feature Importance (Regression Coefficients)")
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Error during prediction")
        st.exception(e)
