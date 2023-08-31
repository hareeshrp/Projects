import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from vit import ViT, ClassToken # Assuming you have the ViT implementation in a file named "vit.py"
from patchify import patchify


hp = {}
hp['image_size'] = 200
hp['num_channels'] = 3
hp['patch_size'] = 25
hp['num_patches'] = (hp['image_size']**2) // (hp['patch_size']**2)
hp['flat_patches_shape'] = (hp['num_patches'], hp['patch_size']*hp['patch_size']*hp['num_channels'])

hp['batch_size'] = 32
hp['lr'] = 1e-4
hp['num_epochs'] = 500
hp['num_classes'] = 4
hp['class_names'] = ['Covid','Lung Opacity', 'Normal', 'Viral Pneumonia']

hp["num_layers"] = 12
hp["hidden_dims"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

# Load the trained model
model_path = "model.h5"
custom_objects = {"ClassToken":ClassToken}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

# Class names
class_names = ['Covid', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

# Streamlit interface
st.title("ViT Image Classifier")
st.write("Upload an image for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (hp['image_size'], hp['image_size']))
    image = image/255.0

    # Divide the image into patches
    patches = patchify(image, (hp['patch_size'], hp['patch_size'], hp['num_channels']), hp['patch_size'])
    patches = np.reshape(patches, (hp['num_patches'], -1))

    # Make a prediction
    predictions = model.predict(np.expand_dims(patches, axis=0))
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write(predictions)

