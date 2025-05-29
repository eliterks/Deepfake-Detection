import os
import numpy as np
from utils import extract_frames_from_video
from model import build_model

DATA_DIR = 'dataset/train'
CATEGORIES = ['real', 'fake']

X = []
y = []

for label, category in enumerate(CATEGORIES):
    path = os.path.join(DATA_DIR, category)
    for file in os.listdir(path):
        if file.endswith('.mp4'):
            video_path = os.path.join(path, file)
            frames = extract_frames_from_video(video_path)
            X.append(frames)
            y.append(label)

X = np.array(X)
y = np.array(y)

model = build_model()
model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

model.save('deepfake_model.h5')
