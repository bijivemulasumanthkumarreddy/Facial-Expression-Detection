import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_frame
from anti_spoofing.spoof_check import is_real_face
from tensorflow.keras.models import model_from_json

import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential  # <-- Explicitly import Sequential
from tensorflow.keras.optimizers import Adam

# Load the model architecture
with open("emotion_model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)  # now Sequential is known

# Load the weights
model.load_weights("emotion_model.h5")

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("😊 Real-Time Emotion Detection")

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.h5")
model = load_emotion_model()
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    try:
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        if not is_real_face(frame):
            st.warning("⚠️ Spoofing Detected!")
        else:
            img = preprocess_frame(frame, size=128)
            preds = model.predict(img)
            emotion = labels[np.argmax(preds)]
            st.success(f"😊 Detected Emotion: **{emotion}**")
            st.image(frame, channels="BGR", caption="Your Photo")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
