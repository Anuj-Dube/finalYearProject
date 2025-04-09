import os
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import streamlit as st
import pickle
import xgboost as xgb
from PIL import Image
import cv2

# Load the trained XGBoost model using pickle
model = pickle.load(open('model.pkl', 'rb'))

# Define the function to predict anemia
def predict_anemia(img_pil, feature_extractor, model, image_size=(224, 224)):
    st.write("Predict anemia")
    # Convert the PIL Image to a numpy array after ensuring it's in RGB mode (drop alpha channel)
    img_np = np.array(img_pil.convert("RGB"))
    # Resize using cv2 and process for the CNN
    img_resized = cv2.resize(img_np, image_size)
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features using the CNN feature extractor
    features = feature_extractor.predict(img_array)

    # Predict using the trained XGBoost model
    dmatrix = xgb.DMatrix(features)
    pred_prob = model.predict(dmatrix)[0]
    pred_label = 1 if pred_prob > 0.5 else 0

    return "Anemic" if pred_label == 1 else "Non-anemic"

# Set up the feature extractor (VGG19 without top layers)
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
gap_output = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=gap_output)

# Streamlit UI for file upload
st.set_page_config(page_title="XGBoost Image Classifier")
st.title("📷 XGBoost Image Classifier")
st.write("Upload an image to get a prediction from the model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image to RGB immediately to avoid 4-channel issues
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Predicting..."):
        preds = predict_anemia(img, feature_extractor, model)
        st.success(f"✅ Prediction: {preds}")
