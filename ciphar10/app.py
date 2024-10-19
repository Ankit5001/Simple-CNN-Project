import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


model = tf.keras.models.load_model(r'C:\Users\Ankit Chaudhary\OneDrive\Desktop\ciphar10\ciphar10Classifier.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(img):
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


st.title("CIFAR-10 Image Classification")
st.write("Upload an image to classify it into one of the CIFAR-10 classes.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    st.subheader(class_names[np.argmax(score)])
    st.subheader(f"confidence_score: =>{100 * np.max(score):.2f}")

