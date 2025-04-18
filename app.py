import streamlit as st
import requests

# Set your FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fertilizer Recommender", layout="centered")
st.title("🌾 Fertilizer Recommendation System")
st.markdown("Fill in the details below to get top 3 fertilizer suggestions.")

# Input fields
temparature = st.number_input("Temperature (°C)", value=0.0)
humidity = st.number_input("Humidity (%)", value=0.0)
moisture = st.number_input("Moisture (%)", value=0.0)
nitrogen = st.number_input("Nitrogen (N)", value=0.0)
phosphorous = st.number_input("Phosphorous (P)", value=0.0)
potassium = st.number_input("Potassium (K)", value=0.0)

# Replace these with values from your actual dataset
soil_types = ["Sandy", "Loamy", "Clayey", "Black", "Red"]
crop_types = ["Maize", "Wheat", "Sugarcane", "Paddy", "Cotton"]

soil_type = st.selectbox("Soil Type", soil_types)
crop_type = st.selectbox("Crop Type", crop_types)

# Button to submit
if st.button("Recommend Fertilizers"):
    payload = {
        "Temparature": temparature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Nitrogen": nitrogen,
        "Phosphorous": phosphorous,
        "Potassium": potassium,
        "Soil_Type": soil_type,
        "Crop_Type": crop_type
    }

    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ Top 3 Fertilizer Recommendations:")
                for i, (fert, prob) in enumerate(zip(result["Top 3 Recommended Fertilizers"], result["Probabilities"]), 1):
                    st.write(f"**{i}. {fert}** — Probability: `{prob:.4f}`")
        except Exception as e:
            st.error(f"Failed to get response from API: {e}")
