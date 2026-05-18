import streamlit as st
import numpy as np
import pickle

# Load model
with open("cancer_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Cancer Classification using AdaBoost")

radius = st.number_input("Mean Radius")
texture = st.number_input("Mean Texture")
perimeter = st.number_input("Mean Perimeter")
area = st.number_input("Mean Area")
smoothness = st.number_input("Mean Smoothness")

if st.button("Predict"):

    input_data = np.array([[
        radius,
        texture,
        perimeter,
        area,
        smoothness
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Benign Cancer")
    else:
        st.error("Malignant Cancer")