import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('return_ann_model_v1.h5')

# Load encoders
with open('label_encoder_gender.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('product_label_encoder.pkl', 'rb') as f:
    product_label_encoder = pickle.load(f)

with open('payment_encoder.pkl', 'rb') as f:
    payment_encoder = pickle.load(f)

with open('shipping_encoder.pkl', 'rb') as f:
    shipping_encoder = pickle.load(f)

with open('return_reason_encoder.pkl', 'rb') as f:
    return_encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit App
st.title("📦 Product Return Prediction")

# Input fields
product = st.selectbox("Product Category", product_label_encoder.classes_)
age = st.slider("User Age", 18, 80)
gender = st.selectbox("User Gender", gender_encoder.classes_)
payment_method = st.selectbox("Payment Method", payment_encoder.categories_[0])
shipping_method = st.selectbox("Shipping Method", shipping_encoder.categories_[0])
return_reason = st.selectbox("Return Reason", return_encoder.categories_[0])

# Encoding
product_encoded = product_label_encoder.transform([product])[0]
gender_encoded = gender_encoder.transform([gender])[0]
payment_encoded = payment_encoder.transform([[payment_method]])
shipping_encoded = shipping_encoder.transform([[shipping_method]])
return_encoded = return_encoder.transform([[return_reason]])

# Final input array
input_array = np.array([[product_encoded, age, gender_encoded]])
input_encoded = np.concatenate([input_array, payment_encoded, return_encoded, shipping_encoded], axis=1)

# Scaling
input_scaled = scaler.transform(input_encoded)

# Prediction
prediction = model.predict(input_scaled)[0][0]

# Output
if prediction > 0.5:
    st.error(f"🔁 This product is likely to be returned. (Confidence: {prediction:.2f})")
else:
    st.success(f"✅ This product is not likely to be returned. (Confidence: {1 - prediction:.2f})")

