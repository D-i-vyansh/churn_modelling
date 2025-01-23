import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("Customer Churn Prediction")

model = load_model("model.h5")
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gen = pickle.load(file)
with open("onehot.pkl", "rb") as file:
    one_hot_geo = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

geography = st.selectbox("Geography", one_hot_geo.categories_[0])  
gender = st.selectbox("Gender", label_encoder_gen.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gen.transform([gender])[0]],  # Fix typo
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_geo.get_feature_names_out(["Geography"]))


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = scaler.transform(input_data)


prediction = model.predict(input_data)
prediction_proba = prediction[0][0]

if prediction_proba >= 0.5:
    st.write("The customer will churn.")
else:
    st.write("The customer will not churn.")
