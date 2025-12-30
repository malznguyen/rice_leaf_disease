# 🌾 Rice Leaf Disease Detection

A deep learning project for detecting and classifying diseases in rice leaves using state-of-the-art object detection models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![Detectron2](https://img.shields.io/badge/Detectron2-Facebook-purple.svg)

## 📋 Overview

This project implements multiple object detection algorithms to identify and classify rice leaf diseases. Early detection of plant diseases is crucial for agricultural productivity and food security.

### Disease Classes

The model detects **4 types of rice leaf diseases**:

| Disease | Vietnamese Name | Description |
|---------|----------------|-------------|
| 🔴 **Bacterial Blight** | Bạc lá | Caused by *Xanthomonas oryzae*, leads to yellowing and wilting |
| 🟠 **Blast** | Đạo ôn | Caused by *Magnaporthe oryzae*, creates diamond-shaped lesions |
| 🟤 **Brown Spot** | Đốm nâu | Fungal disease causing circular brown spots |
| 🟡 **Tungro/Twisted Draft** | Vàng lùn | Viral disease causing stunting and leaf discoloration |

## 🏗️ Project Structure

```
DeepLearning/
├── notebooks/
│   ├── YOLO_RLD_F.ipynb          # YOLOv8 Oriented Bounding Box model
│   ├── Faster_Rcnn-RLD-F.ipynb   # Faster R-CNN with Detectron2
│   └── SSD_RLD_F.ipynb           # SSD (Single Shot Detector)
├── exp-obb-100-*/                 # YOLO training results
│   └── exp-obb-100/
│       ├── weights/               # Trained model weights
│       ├── confusion_matrix.png   # Confusion matrix visualization
│       ├── F1_curve.png          # F1 score curve
│       ├── PR_curve.png          # Precision-Recall curve
│       ├── results.csv           # Training metrics per epoch
│       └── ...
├── test_results-obb-100-*/        # Test evaluation results
├── data.yaml                      # Dataset configuration
└── README.md
```

## 🛠️ Models Implemented

### 1. YOLOv8-OBB (Oriented Bounding Box)
- **Architecture**: YOLOv8n-OBB
- **Training**: 100 epochs
- **Image Size**: 640x640
- **Batch Size**: 48
- **Optimizer**: Auto (AdamW)

### 2. Faster R-CNN
- **Backbone**: ResNet-50 FPN
- **Framework**: Detectron2 (Facebook Research)
- **Max Iterations**: 3,000
- **Batch Size**: 4
- **Learning Rate**: 0.00025

### 3. SSD (Single Shot Detector)
- Custom SSD implementation for rice disease detection

## 📊 Results

### YOLOv8-OBB Performance (100 epochs)

| Metric | Value |
|--------|-------|
| **mAP@50** | 89.75% |
| **mAP@50-95** | 73.80% |
| **Precision** | 90.54% |
| **Recall** | 86.24% |

### Training Progress

The model shows consistent improvement over 100 epochs:
- Box Loss: 1.84 → 0.80
- Classification Loss: 4.31 → 0.68
- DFL Loss: 3.10 → 1.62

## 🚀 Quick Start

### Prerequisites

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# For YOLOv8
pip install ultralytics

# For Faster R-CNN (Detectron2)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Dataset Structure

```
RiceLeafDisease/
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
```

### Training

#### YOLOv8
```python
from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=48
)
```

#### Faster R-CNN
```python
# Run the Jupyter notebook
# notebooks/Faster_Rcnn-RLD-F.ipynb
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('path/to/best.pt')

# Run inference
results = model('rice_leaf_image.jpg')
results[0].show()
```

## 📈 Visualizations

The training process generates various visualizations:

- **Confusion Matrix**: Shows prediction accuracy per class
- **F1 Curve**: F1 score at different confidence thresholds
- **Precision-Recall Curve**: Trade-off between precision and recall
- **Training Batches**: Sample training images with annotations
- **Validation Predictions**: Comparison of ground truth vs predictions

## 🔧 Configuration

### data.yaml
```yaml
train: /path/to/RiceLeafDisease/train/images
val: /path/to/RiceLeafDisease/valid/images
test: /path/to/RiceLeafDisease/test/images
nc: 4
names: ['Bacterial Blight', 'Blast', 'Brown Spot', 'Twisted Draft']
```

## 📚 References

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Detectron2 by Facebook Research](https://github.com/facebookresearch/detectron2)
- [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets)

## 👥 Contributors

- Project developed for MS-2025 Deep Learning course

## 📄 License

This project is for educational purposes.

---

<p align="center">
  Made with ❤️ for Agricultural AI Research
</p>
