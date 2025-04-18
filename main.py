from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize app
app = FastAPI(title="Fertilizer Recommendation API")

# Load models
model: XGBClassifier = joblib.load("fertilizer_model.pkl")
scaler: StandardScaler = joblib.load("scaler.pkl")
kmeans_cat: KMeans = joblib.load("cluster_model_cat.pkl")
kmeans_num: KMeans = joblib.load("cluster_model_num.pkl")

# Load encoders
le_soil = joblib.load("le_soil.pkl") if os.path.exists("le_soil.pkl") else None
le_crop = joblib.load("le_crop.pkl") if os.path.exists("le_crop.pkl") else None
le_fert = joblib.load("le_fert.pkl") if os.path.exists("le_fert.pkl") else None

# Input data model
class FertilizerInput(BaseModel):
    Temparature: float
    Humidity: float
    Moisture: float
    Nitrogen: float
    Phosphorous: float
    Potassium: float
    Soil_Type: str
    Crop_Type: str

@app.post("/predict")
def predict_fertilizer(data: FertilizerInput):
    try:
        # Encode soil and crop
        soil_encoded = le_soil.transform([data.Soil_Type])[0]
        crop_encoded = le_crop.transform([data.Crop_Type])[0]

        # Scale numeric inputs
        num_input = np.array([[data.Temparature, data.Humidity, data.Moisture,
                               data.Nitrogen, data.Phosphorous, data.Potassium]])
        num_scaled = scaler.transform(num_input)

        # Interaction terms
        temp_humidity = data.Temparature * data.Humidity
        moisture_nitrogen = data.Moisture * data.Nitrogen

        # Cluster features
        cat_cluster = kmeans_cat.predict([[soil_encoded, crop_encoded, 0]])[0]
        num_cluster = kmeans_num.predict([[data.Nitrogen, data.Phosphorous, 0]])[0]

        # Combine features
        final_features = np.concatenate([
            num_scaled[0],
            [soil_encoded, crop_encoded, cat_cluster, num_cluster, temp_humidity, moisture_nitrogen]
        ]).reshape(1, -1)

        # Predict top 3 fertilizers
        proba = model.predict_proba(final_features)
        top3_idx = np.argsort(proba[0])[::-1][:3]
        top3_ferts = le_fert.inverse_transform(top3_idx)

        return {
            "Top 3 Recommended Fertilizers": top3_ferts.tolist(),
            "Probabilities": [float(proba[0][i]) for i in top3_idx]
        }

    except Exception as e:
        return {"error": str(e)}
