# 🌾 Fertilizer Recommendation System

This project is designed to recommend the most suitable fertilizer based on crop type, soil type, and environmental conditions using machine learning. The goal is to help farmers optimize fertilizer use for better crop yield and soil health.

---

## 🚀 Features

- ✅ Predicts the best fertilizer based on input conditions
- ✅ Uses a trained ML classification model
- ✅ Web-based interface using FastAPI
- ✅ Includes preprocessing with clustering and feature encoding
- ✅ Supports categorical and numerical soil/crop features

---
The repository contains the following key files:​

app.py: Main application script that runs the web interface.

main.py: Script containing the core logic for processing inputs and generating recommendations.

crop and soil.ipynb: Jupyter notebook for exploratory data analysis and model development.

crop_recommendation.ipynb: Notebook focusing on crop recommendation logic.

fertilizer_model.pkl: Serialized machine learning model for fertilizer prediction.

cluster_model_cat.pkl & cluster_model_num.pkl: Clustering models for categorical and numerical data.

le_crop.pkl, le_fert.pkl, le_soil.pkl: Label encoders for crops, fertilizers, and soil types.

scaler.pkl: Feature scaler used during model training.

requirement.txt: List of Python dependencies required to run the project

## 🧠 Machine Learning

The model is trained on a custom dataset containing:

- Environmental parameters: Temperature, Humidity, Moisture
- Soil characteristics: Soil Type (categorical), Nutrient levels (N, P, K)
- Crop Type (categorical)
- Target: Fertilizer Name

Models used:
- XGBoost Classifier (with clustering and scaling)
- Label Encoders and Scalers for consistent prediction

---

# See my full end-to-end Machine learning Deployment :- 
https://fertilizer-recommendation-system-3kufnbucriptzw7hg3ktzm.streamlit.app/

