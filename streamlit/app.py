import streamlit as st
from xgboost import XGBClassifier
import joblib
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
from json import JSONDecodeError
import re
import time
import os
from openai import OpenAI

#-- ACTIVATING AI KEYS
API_KEY= st.secrets["GROQ_API_KEY"]
client= OpenAI(
    base_url= "https://api.groq.com/openai/",
    api_key= API_KEY
)

#-- CLASS FOR MODEL FEATURES
class ModelFeatures(BaseModel):
    longitude: float
    latitude: float
    srf_salt: int =Field #(alias= "salt surface")
    srf_crust: int =Field #(alias= "crust surface")
    bare_soil: int =Field #(alias= "bare soil")
    altitude: float =Field

#-- THE PREDICT FUNCTION
def predict_from_features(features: ModelFeatures) -> str:
    #coords= np.array([[features.latitude, features.longitude]])
    coords= pd.DataFrame([[features.longitude, features.latitude]], columns= ['longitude', 'latitude'])
    cluster= spatial_mapper.predict(coords)
    encoded_cluster= cluster_encoder.transform(cluster.reshape(1, -1))
    scaled_altitude= altitude_scaler.transform([[features.altitude]])

    row= pd.DataFrame([{
        "longitude": features.longitude,
        "latitude": features.latitude,
        "srf_salt": features.srf_salt,
        "srf_crust": features.srf_crust,
        "bare_soil": features.bare_soil,
        "geo_clusters": encoded_cluster[0][0] if len(encoded_cluster.shape) > 1 else encoded_cluster[0],
        "scaled_altitude": scaled_altitude[0][0]
    }])

    prediction= model.predict(row)
    pred_encoded= label_encoder.inverse_transform(prediction)
    return pred_encoded[0]

#-- FUNCTION TO EXTRACT JSON FROM MODEL REPLY
def extract_json(reply):
    features= re.search(r'\{.*\}', reply, re.DOTALL)
    if features:
        return json.loads(features.group(0))
    else:
        raise JSONDecodeError

#-- FUNCTION TO "STREAM" TEXT
def stream_text(text, delay= 0.04):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)

#-- TO GET ABSOLUTE PATH
BASE_DIR= os.path.dirname(os.path.abspath(__file__))

model_path= os.path.join(BASE_DIR, "xgbmodel.pkl")
altitude_scaler_path= os.path.join(BASE_DIR, "altitude_scaler.pkl")
cluster_encoder_path= os.path.join(BASE_DIR, "cluster_encoder.pkl")
label_encoder_path= os.path.join(BASE_DIR, "prediction_label_encoder.pkl")
spatial_mapper_path= os.path.join(BASE_DIR, "knn_spatial_mapper.pkl")

#-- LOADING NECCESSARY ARTIFACTS
model = joblib.load(model_path)
altitude_scaler= joblib.load(altitude_scaler_path)
cluster_encoder= joblib.load(cluster_encoder_path)
label_encoder= joblib.load(label_encoder_path)
spatial_mapper= joblib.load(spatial_mapper_path)

#-- INITIALISING SESSION STATES
if "phase" not in st.session_state:
    st.session_state.phase= "gathering"
    st.session_state.predicted_crop= None

    st.session_state.messages= [
        {"role": "system", "content": """You are a soil analyst assistant. Sound natural and conversational.
    Ask the user for the following information one by one:
    - The city or location (Use this to secretly estimate longitude and latitude, do not ask for coordinates directly)
    - If there is a noticeable salt layer on the surface
    - If there is a crust forming on the surface
    - If there is bare soil (no vegetation)
    - The altitude in meters

    The integer valued fields for salt, crust, and bare soil are simply 0 and 1. If the user says yes, then 0. If no, then 1. Let their natural answers guide you.

    CRITICAL JSON INSTRUCTIONS:
    Once you have all values, reply with ONLY a valid, flat JSON object. Do NOT nest it.
    Format exactly like this example:
    {{
        "longitude": 39.2026,
        "latitude": -6.7924,
        "srf_salt": 0,
        "srf_crust": 1,
        "bare_soil": 0,
        "altitude": 340
    }}

    No extra text, just the JSON. Do not explain the backend process.
    """},
        {"role": "assistant", "content": "Hi! I'm your AI soil analyst. To figure out the best crop, what city or region is the farm located in?"}
    ]

#-- RENDER CHAT UI
st.title("Nathan's Soil Predictor Model v1.0")
st.caption("Soil Predictor ML Model powered by Agentic AI and XGBoost")

#-- DISPLAYING ALL MESSAGES EXCEPT SYSTEM PROMPTS
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

#-- HANDLING USER INPUT
if user_input:= st.chat_input("Reply to agent"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

#-- CALLING THE LLM
with st.chat_message("assistant"):
    response= client.chat(model= "gemma4:31b-cloud", messages= st.session_state.messages)
    reply= response.message.content

    #-- PARSING JSON
    if st.session_state.phase== "gathering":
        match= re.search(r"\{.*\}", reply, re.DOTALL)
        if match:
            #--- RUNNING XGB
            data= json.loads(match.group(0))
            features= ModelFeatures(**data)
            predicted_crop= predict_from_features(features)
            st.session_state.predicted_crop= predicted_crop

            #-- PHASE CHANGE
            st.session_state.phase= "qa"
            new_system_prompt= f'''You are now an expert agricultural advisor. 
            Based on the soil data, the best crop to plant is {predicted_crop}.
            Answer any questions the user has about planting, fertilizing, or harvesting {predicted_crop}.
            Keep your tone encouraging and helpful.'''
            st.session_state.messages[0]= {"role":"system", "content": new_system_prompt}
            transition_msg= f"Analysis complete!, Based on the information you provided, the besst crop for the land is {predicted_crop} \
            feel free to ask if you have any questions about planting, fertilizing, or harvesting {predicted_crop}"
            st.write_stream(stream_text(transition_msg))
            st.session_state.messages.append({"role": "assistant", "content": transition_msg})
        else:
            #-- IF JSON NOT EXTRACTED, PHASE 1 CHAT PROCEEDS
            st.write_stream(stream_text(reply))
            st.session_state.messages.append({"role": "assistant", "content": reply})
    
    #-- PHASE 2 CHAT
    elif st.session_state.phase== "qa":
        st.write_stream(stream_text(reply))
        st.session_state.messages.append({"role": "assistant", "content": reply})
