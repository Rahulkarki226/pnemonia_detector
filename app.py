import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("PD.h5")


# Define the function to preprocess the uploaded image
def preprocess_image(image):
    image = np.array(image)
    if len(image.shape) == 2:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    image = cv2.resize(image, (220, 220))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Define the Streamlit app

st.title("Pneumonia Detection from Chest X-ray Images")
st.write("Upload a chest X-ray image to predict if it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    st.write("")
    st.write("Classifying...")

    image = preprocess_image(image)
    prediction = model.predict(image)

    if prediction >= 0.5:
        st.write("The X-ray image shows signs of **Pneumonia**.")
    else:
        st.write("The X-ray image shows **Normal** lungs.")

