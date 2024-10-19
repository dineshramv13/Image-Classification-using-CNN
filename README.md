This project implements image classification using deep learning techniques, comparing the performance of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN). A baseline ANN model is developed and evaluated for classifying images into predefined categories. To enhance accuracy, a CNN architecture is constructed, leveraging spatial relationships within images. The CNN significantly outperforms the ANN, demonstrating its strength in capturing intricate visual patterns. Additionally, the project includes preprocessing and classifying custom images using the trained CNN model. The project is deployed using Streamlit for a user-friendly interface that allows image upload and classification.

Key Features
Baseline ANN model and advanced CNN for image classification.
Preprocessing pipeline for custom image classification.
Streamlit deployment for easy access and interaction.


Setup

pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py


Results
The CNN model outperforms the ANN baseline, effectively classifying custom images and demonstrating the practical utility of deep learning in real-world image classification tasks.

Future Scope
Future work includes hyperparameter tuning, increasing dataset diversity, and extending the application to real-time image classification on web or mobile platforms.

