# 🧠 Deepfake Detection using InceptionV3 + GRU

This repository contains a lightweight deepfake video classification system using a hybrid architecture of **InceptionV3** and **GRU**. It was built specifically for systems with **CPU-only environments**, such as Intel integrated graphics, as part of the **AIMS-DTU Research Intern Round 2**.

---

## 📁 Dataset Structure

The dataset should be organized as follows, with `.mp4` video files in each folder:

dataset/
├── train/
│ ├── real/
│ └── fake/
└── test/
├── real/
└── fake/


---

## ⚙️ Tech Stack

- Python 3.10  
- TensorFlow 2.10  
- OpenCV  
- NumPy, Matplotlib, Scikit-learn  
- InceptionV3 (pretrained on ImageNet)  
- GRU (Gated Recurrent Unit for temporal modeling)  

---

## 🧠 Model Architecture

Video (10 frames)  
→ TimeDistributed(InceptionV3)  
→ TimeDistributed(GlobalAveragePooling2D)  
→ GRU (64 units)  
→ Dense (64 units, relu)  
→ Dense (1 unit, sigmoid) → Real or Fake

> InceptionV3 is frozen and used only as a spatial feature extractor. GRU captures temporal consistency across frames.

---

## 🛠️ How It Works

### Preprocessing

- Extract 10 evenly spaced frames per video using OpenCV.  
- Resize frames to 224x224 pixels.  
- Normalize pixel values.

### Training

Run the training script:

```bash
python train.py

Training details:

- Optimizer: Adam

- Loss: Binary Crossentropy

- Epochs: 5

- Batch Size: 2

Inference
Classify a new video as real or fake:
python predict.py dataset/test/fake/sample.mp4

📊 Results (on Intel i7 CPU)
| Metric         | Value                   |
| -------------- | ----------------------- |
| Accuracy       | 85%                     |
| AUC Score      | 0.87                    |
| Inference Time | \~1.8 seconds per video |

📂 Files in this Repository

| File                | Description                                |
| ------------------- | ------------------------------------------ |
| `train.py`          | Trains the InceptionV3 + GRU model         |
| `predict.py`        | Classifies new video as real or fake       |
| `model.py`          | Model definition (CNN + GRU)               |
| `utils.py`          | Frame extraction and preprocessing helpers |
| `requirements.txt`  | Python dependencies                        |
| `Documentation.pdf` | Final project documentation (AIMS-DTU)     |

🔮 Future Improvements
- Integrate face alignment / landmark detection to improve accuracy.

- Replace InceptionV3 with MobileNet or EfficientNet for even lower latency.

- Explore video transformers (e.g., ViViT, TimeSformer) once GPU resources are available.

- Enable real-time detection for streaming content.

🤝 Contributions
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.