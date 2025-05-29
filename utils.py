import cv2
import numpy as np
import os

def extract_frames_from_video(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)

    for i in range(max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (299, 299))
        frames.append(frame)
    
    cap.release()
    return np.array(frames)
