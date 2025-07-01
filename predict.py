# Import standard libraries
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Import Streamlit for app interface
import streamlit as st

# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow_hub.keras_layer import KerasLayer  # If using TF Hub layers

# Inisialisasi class labels
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Fungsi prediksi
def import_and_predict(image_data, model, class_names=class_names):
    image = load_img(image_data, target_size=(227, 227))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    predicted_label = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100  # dalam persen

    return f"Prediction: {predicted_label} (Confidence: {confidence:.2f}%)"

# Fungsi utama
def run():
    st.title("ASL Hand Sign Classification (A–Y)")

    img = Image.open("pred.jpg")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption=" ", use_container_width=True)

    # Pilihan metode input
    input_method = st.radio("Choose the picture taking method:", ["Upload from device", "Take from camera"])

    if input_method == "Upload from device":
        file = st.file_uploader("Upload an image", type=["jpg", "png"])
    else:
        file = st.camera_input("Take from camera")

    if file is None:
        st.info("Please upload or take picture first.")
        return

    model = load_model('modelcnnringan.keras', custom_objects={'KerasLayer': KerasLayer})
    st.image(file, caption="Uploaded Image", use_container_width=True)
    result = import_and_predict(file, model)
    st.success(result)

if __name__ == "__main__":
    run()