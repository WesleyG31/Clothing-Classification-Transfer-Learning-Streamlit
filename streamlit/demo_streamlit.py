import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load the model   
@st.cache_resource
def cargar_modelo(modelo_seleccionado):
    if modelo_seleccionado == "MobileNetV2 without weights":
        return tf.keras.models.load_model("models/model_MobileNetV2_1.h5")
    elif modelo_seleccionado == "Custom CNN":
        return tf.keras.models.load_model("models/model_custom_cnn.h5")

# Models available
modelos_disponibles = ["MobileNetV2 without weights", "Custom CNN"]

# interface configuration
st.title("🧥 Clothes clasification - Fashion MNIST 👕👖👟")
st.write("Select a model and upload an image to classify the type of clothing.")
st.write("### T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot")
st.write("The models were trained with the Fashion MNIST dataset.")
st.write("#### MobileNetV2 without weights = 82.98% accuracy")   # Accuracy del modelo
st.write("#### Custom CNN =  91.07% accuracy")   # Accuracy del modelo

# select model
modelo_seleccionado = st.selectbox("🔍 Chosse the model - Transfer Learning:", modelos_disponibles)

# load model based on selection
modelo = cargar_modelo(modelo_seleccionado)

# classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# upload image
uploaded_file = st.file_uploader("📂 Upload a image...", type=["png", "jpg", "jpeg"])

if uploaded_file:

    # show images uploaded
    st.image(uploaded_file, caption="📷 Image loaded", use_container_width=True)

    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    image = image.resize((32, 32))  # resize to 32x32 pixels
    
    
    
    # preprocessing
    image = np.array(image) / 255.0  # Normalization   
    image = np.expand_dims(image, axis=-1)  # extra channel dimension
    image = np.repeat(image, 3, axis=-1)  # To RGB
    image = np.expand_dims(image, axis=0)  # add batch dimension
    
    
    # Prediction
    prediction = modelo.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # show results
    st.write(f"### ✅ Model: {modelo_seleccionado}")
    st.write(f"### 🏷️ Prediction: {class_names[predicted_class]}")
    st.write(f"### 📊 Confidence: {confidence:.2f}%")
