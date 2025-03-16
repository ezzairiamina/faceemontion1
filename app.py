import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# Fonction pour charger le modèle
@st.cache_resource
def load_model_once():
    try:
        model = tf.keras.models.load_model("emotiondetector.h5")# Charger le modèle .h5 directement
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

# Charger le modèle
model = load_model_once()
if model is None:
    st.stop()  # Stop l'exécution si le modèle ne charge pas

# Charger le détecteur de visages OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

