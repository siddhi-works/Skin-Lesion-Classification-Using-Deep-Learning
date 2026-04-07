import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import gdown
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "skin_cancer_cnn.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1o9yTKrFX9UyO1c0thpwrxoyV66fXLagt"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img

# UI APP CODE
st.title("AI-Powered Dermoscopic Skin Lesion Detection & Image Analysis")

st.markdown("""
An AI-powered diagnostic system for skin lesion classification using Convolutional Neural Networks (CNN). 
The model analyzes dermoscopic images and provides instant predictions to assist in early detection.
""")

st.subheader("How to use:")
st.markdown("""
1. Upload a skin lesion image  
2. Wait for prediction  
3. View the result  
""")

uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    class_label, img = predict_skin_cancer(uploaded_image, model)

    st.image(uploaded_image, caption='Uploaded image', width=500)
    st.write("Prediction :", class_label)

st.subheader("About the Model:")
st.write("""
This model uses CNN architecture for predicting whether a skin lesion is Benign or Malignant based on images of skin lesions.
""")

st.subheader("Features:")
st.markdown("""
- Input: Skin lesion images  
- Output: Benign or Malignant classification  
""")

st.subheader("Disclaimer:")
st.warning("""
This application is for educational and demonstration purposes only. 
It is not intended to replace professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional for any medical concerns.
""")

st.markdown("""
****Developed by Siddhi Prasad Kale****
""")
