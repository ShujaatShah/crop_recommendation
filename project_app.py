import streamlit as st
import joblib
import numpy as np
import pandas as pd

model_rf = joblib.load('random_forest_model.pkl')


# Sidebar
st.sidebar.title('Crop Recommendation App')
st.sidebar.markdown(
    "Welcome to our Crop Recommendation App! This tool recommends crops based on environmental factors such as nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall."
)


# Main content
st.title('Crop Recommendation App')

# User input for feature values
st.subheader('Enter Environmental Conditions:')
n = st.slider('Select the number of crops to recommend:', 1, 4, 2)
nitrogen = st.number_input('Nitrogen (0-150)',min_value=0, max_value=150, value=50)
phosphorus = st.number_input('Phosphorus (5-150)', min_value=5, max_value=150, value=50)
potassium = st.number_input('Potassium (0-220)', min_value=0, max_value=220, value=50)
temperature = st.number_input('Temperature (0-50)', min_value=0, max_value=50, value=25)
humidity = st.number_input('Humidity (5-100)', min_value=5, max_value=100, value=50)
ph = st.slider('pH (0-14)', min_value=0, max_value=14, value=7)
rainfall = st.number_input('Rainfall (0-300)', min_value=0, max_value=300, value=150)

# Predicting crops based on user input
user_input = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
prob_total = model_rf.predict_proba(user_input)
n_indices = prob_total[0].argsort()[-n:][::-1]
top_n_crops = [model_rf.classes_[i] for i in n_indices]

# Display recommended crops
st.subheader(f'Top {n} Recommended Crops:')
for i in top_n_crops:
    st.write(i)


