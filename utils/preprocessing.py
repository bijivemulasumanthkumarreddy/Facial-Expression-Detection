import cv2
import numpy as np

def preprocess_frame(frame, size=128):
    face = cv2.resize(frame, (size, size))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)
    return face
