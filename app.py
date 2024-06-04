import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model
model = load_model('CNN.h5')

# Define your class names
class_names =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("Image Classification with CNN")
st.write("Upload an image and the model will predict its class.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
 


    img = img.resize((32, 32))  # Resize the image to 28x28
    img = np.array(img)  # Convert to numpy array
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    

    # Predict the class
    predictions = model.predict(img)

    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]

    st.write(f"Prediction: {predicted_class}")
    #st.write(f"Confidence: {100 * np.max(score):.2f}%")

