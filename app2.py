import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your models
cnn_model = load_model('CNN.h5')
ann_model = load_model('ANN.h5')

# Define your class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Image Classification with CNN and ANN")
st.write("Upload an image and the models will predict its class.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((32, 32))  # Resize the image to 32x32
    img = np.array(img)  # Convert to numpy array
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class using CNN
    cnn_predictions = cnn_model.predict(img)
    cnn_score = tf.nn.softmax(cnn_predictions[0])
    cnn_predicted_class = class_names[np.argmax(cnn_score)]

    # Predict the class using ANN
    ann_predictions = ann_model.predict(img)
    ann_score = tf.nn.softmax(ann_predictions[0])
    ann_predicted_class = class_names[np.argmax(ann_score)]

    # Display the predictions
    col1, col2 = st.columns(2)

    with col1:
        st.header("CNN Model Prediction")
        st.write(f"Prediction: {cnn_predicted_class}")
        st.write(f"Confidence: {100 * np.max(cnn_score):.2f}%")

    with col2:
        st.header("ANN Model Prediction")
        st.write(f"Prediction: {ann_predicted_class}")
        st.write(f"Confidence: {100 * np.max(ann_score):.2f}%")
