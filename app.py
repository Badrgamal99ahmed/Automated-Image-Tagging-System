import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Load your pre-trained model
model = tf.keras.models.load_model('caption_model.keras')

# Define function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0,1]
    return image


# Placeholder function for decoding caption (modify according to your model's structure)
def decode_caption(tokens):
    # This function should convert tokens or indices to a human-readable caption
    return " ".join([str(token) for token in tokens])

# Define function to generate caption
def generate_caption(image):
    # Preprocess image
    preprocessed_image = preprocess_image(image, (224, 224))  # Use your model's input size
    # Generate caption (assuming model.predict returns caption tokens)
    caption_tokens = model.predict(preprocessed_image)
    # Convert tokens to readable caption (this depends on your model's output format)
    caption = decode_caption(caption_tokens)
    return caption



# Streamlit app layout
st.title('Image Captioning App')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Generate caption
    st.write("Generating caption...")
    caption = generate_caption(image)

    # Display the result
    st.write(f"Caption: {caption}")