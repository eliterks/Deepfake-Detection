import sys
from utils import extract_frames
from model import build_model
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('deepfake_model.h5')

video_path = sys.argv[1]  # Pass path from terminal
frames = extract_frames(video_path)
frames = np.expand_dims(frames, axis=0)
pred = model.predict(frames)[0][0]

print("Prediction:", "Fake" if pred > 0.5 else "Real")
