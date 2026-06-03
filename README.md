# Nathan's Soil Predictor Model v1.0

## Overview
Nathan's Soil Predictor is a Machine Learning model powered by an XGBoost backend and an Agentic AI frontend. It is designed to act as an AI agricultural advisor, estimating the most suitable crop to plant on a given plot of land based on environmental and geographic conditions.

## Capabilities
The model leverages a conversational AI assistant to gather context from the user and estimates the following parameters:
- **Location Context**: Longitude and latitude (derived from the user's city or region).
- **Altitude**: Altitude of the location in meters.
- **Surface Conditions**: Presence of a noticeable salt layer, surface crust formation, and bare soil.

These inputs are mapped geographically (using a k-nearest neighbors spatial mapper) and then fed into an XGBoost classifier to output the optimal crop choice. After the prediction, the Agentic AI can answer follow-up agricultural questions about planting, fertilizing, or harvesting the predicted crop.

## Flaws and Limitations (PLEASE READ)
While the model provides AI-driven crop suggestions, users should be aware of the following flaws and limitations:
- **Biased Data**: The model is trained on a specific dataset of soil and crop plots (`plot_field_data_download.csv`). Its geographical awareness (clusters and spatial mapping) is strictly bound to the locations present in the training data. If your region is not well-represented in the dataset, the predictions may be inaccurate or highly biased.
Additionally, dominance of a certain crop type like Cereals and scarcity of other crop types made training biased, as model has not learnt enough from other crop types. This can be seen in `confusion_matrix.png`.
- **Unreliable fields for prediction**: Fields such as `srf_salt`, `srf_crust` and `bare_soil` may not be the most reliable indicators of crop type, as factors like raifall, temperature, pH were not used during training.
- **Suggestions are not based on the best crop**: This is because the model was trained on a dataset only presented what a farmer chose to plant in a specific area, rather than what the best crop for that piece of land actually is scientifically.
- **Proxy Variables**: The model infers coordinates indirectly through the conversational agent. Misunderstandings by the AI or generic city names can lead to incorrect longitude/latitude assignments.
- **Simplified Features**: The surface condition inputs (salt, crust, bare soil) are treated as binary (0/1) indicators, lacking the nuance of a professional soil laboratory test. 
- **Hallucinations**: The conversational assistant answering follow-up queries relies on an LLM, which may occasionally hallucinate facts about fertilizing or planting.

## ⚠️ WARNING: Use at Your Own Risk
**Disclaimer**: This model is a predictive tool built for experimental and educational purposes. It is **NOT** a substitute for professional agricultural consultation, certified soil laboratory analysis, or expert agronomic advice.

I assume no responsibility for any crop failure, financial loss, or damages resulting from actions taken based on this model's predictions. **By using this software, you agree to use it strictly at your own risk.**