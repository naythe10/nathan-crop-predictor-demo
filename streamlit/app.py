import streamlit as st
import joblib
import agent

#Loading pipeline
model = joblib.load("xgbmodel.pkl")
mapper = joblib.load("knn_spatial_mapper.pkl")
altitude_scaler = joblib.load("altitude_scaler.pkl")
label_encoder = joblib.load("prediction_label_encoder.pkl")

st.title("Crop type predictor")