import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Class names A–Z without J and Z
class_names = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y'
]

# Image preprocessing and prediction
def import_and_predict(image_data, model, target_size=(227, 227)):
    try:
        image = Image.open(image_data).convert("RGB")
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        confidence = predictions[predicted_index] * 100  # convert to percentage
        predicted_class = class_names[predicted_index]

        return predicted_class, confidence
    except Exception as e:
        return f"Error while processing image: {e}", 0

def run():
    img_url = "pred.jpg"
    image = Image.open(img_url)

    # Menampilkan gambar di tengah dengan kolom
    cols = st.columns([1, 2, 1])  # Kiri, tengah, kanan

    with cols[1]:
        st.image(image, use_container_width=True)  # Ganti use_column_width → use_container_width


    st.title("ASL Alphabet Classifier (A–Z without J & Z)")
    input_option = st.radio("Choose input method:", ["Upload", "Camera"])

    if input_option == "Upload":
        file = st.file_uploader("Upload a hand sign image", type=["jpg", "png"])
    else:
        file = st.camera_input("Take a photo using your webcam")

    if not file:
        st.info("Please provide an image above.")
        return

    try:
        model = load_model('modelcnnringan.h5')
    except OSError as e:
        st.error(f"Could not load model: {e}")
        return

    with st.spinner("🔍 Predicting..."):
        prediction, confidence = import_and_predict(file, model)

    st.image(file, caption="📸 Uploaded Image", use_container_width=True)
    st.success(f"🔤 Predicted ASL Letter: **{prediction}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    run()