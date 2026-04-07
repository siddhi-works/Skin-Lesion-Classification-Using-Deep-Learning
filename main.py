import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "skin_cancer_cnn.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1o9yTKrFX9UyO1c0thpwrxoyV66fXLagt"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = prediction[0][0]

    class_label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob if prob > 0.5 else (1 - prob)

    return class_label, confidence, img

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
    class_label, confidence, img = predict_skin_cancer(uploaded_image, model)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_image, caption='Uploaded image', width=300)

    with col2:
        if class_label == "Malignant":
            st.error(f"Prediction: {class_label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            st.warning("This may indicate a serious condition. Please consult a qualified dermatologist as soon as possible.")
            st.markdown("This is a precautionary note provided in the interest of safety.")
        else:
            st.success(f"Prediction: {class_label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")
            st.markdown("Looks harmless, but getting it checked is always a smart move.")

st.subheader("About the Model:")

st.write("""
This model uses CNN architecture for predicting whether a skin lesion is Benign or Malignant based on images of skin lesions.
""")

st.subheader("Features:")

st.markdown("""
- Input: Skin lesion images  
- Output: Benign or Malignant classification with Confidence score 
""")

st.subheader("Disclaimer:")

st.warning("""
This application is for educational and demonstration purposes only. 
It is not intended to replace professional medical advice, diagnosis, or treatment. 
Always consult a qualified healthcare professional for any medical concerns.
""")

st.markdown("""
****Developed with ❤️ by Siddhi Prasad Kale****
""")
