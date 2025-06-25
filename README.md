# Deepfake Detection Project

A machine learning project for detecting deepfake videos using computer vision and deep learning techniques. This project implements a video-based deepfake detection system using InceptionV3 feature extraction and GRU-based sequence modeling.

## 📋 Project Overview

This project addresses the challenge of detecting deepfake videos by analyzing video frames and extracting temporal features. The system uses a two-stage approach:
1. **Feature Extraction**: Using pre-trained InceptionV3 to extract features from video frames
2. **Sequence Modeling**: Using GRU (Gated Recurrent Unit) networks to analyze temporal patterns

## 🎯 Problem Statement

Deepfake technology has become increasingly sophisticated, making it crucial to develop reliable detection methods. This project aims to:
- Classify videos as either REAL or FAKE
- Handle video sequences with varying lengths
- Provide real-time detection capabilities

## 📊 Dataset Analysis

### Dataset Statistics
- **Total Videos**: 400
- **FAKE Videos**: 323 (80.75%)
- **REAL Videos**: 77 (19.25%)
- **Class Imbalance**: Significant imbalance with more fake videos than real ones

### Dataset Structure
The dataset contains:
- Video files in MP4 format
- Metadata JSON file with labels and split information
- Training and test splits for model evaluation

## 🏗️ Architecture

### 1. Feature Extraction Pipeline
```python
# InceptionV3-based feature extractor
- Input: Video frames (224x224x3)
- Output: 2048-dimensional feature vectors
- Pre-trained on ImageNet for robust feature representation
```

### 2. Sequence Modeling
```python
# GRU-based sequence model
- Input: Frame features (20, 2048)
- Hidden layers: GRU(16) → GRU(8) → Dropout(0.4) → Dense(8)
- Output: Binary classification (REAL/FAKE)
```

### 3. Model Parameters
- **Image Size**: 224x224 pixels
- **Max Sequence Length**: 20 frames
- **Feature Dimensions**: 2048
- **Batch Size**: 64
- **Epochs**: 15

## 🚀 Key Features

### Video Processing
- **Center Square Cropping**: Ensures consistent aspect ratios
- **Frame Resizing**: Standardizes frame dimensions
- **Color Channel Reordering**: BGR to RGB conversion
- **Sequence Padding**: Handles videos of varying lengths

### Model Architecture
- **Transfer Learning**: Leverages pre-trained InceptionV3
- **Temporal Modeling**: GRU layers capture sequential patterns
- **Regularization**: Dropout layers prevent overfitting
- **Binary Classification**: Sigmoid activation for final prediction

## 📈 Performance Metrics

The model achieves:
- **Training Accuracy**: ~80.83%
- **Validation Accuracy**: ~80.00%
- **Loss**: Binary cross-entropy optimization

## 🛠️ Technical Implementation

### Dependencies
```python
import cv2
import tensorflow as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from sklearn.model_selection import train_test_split
```

### Core Functions

#### 1. Video Loading
```python
def load_video(path, max_frames=0, resize=(224, 224)):
    # Loads video frames with preprocessing
    # Returns: numpy array of processed frames
```

#### 2. Feature Extraction
```python
def pretrain_feature_extractor():
    # Creates InceptionV3-based feature extractor
    # Returns: Keras model for feature extraction
```

#### 3. Data Preparation
```python
def prepare_all_videos(df, root_dir):
    # Processes all videos for training
    # Returns: frame features, masks, and labels
```

#### 4. Prediction Pipeline
```python
def sequence_prediction(path):
    # Predicts class for single video
    # Returns: prediction probability
```

## 📁 Project Structure

```
Deepfake-detection/
├── Dataset_Metadata.ipynb    # Dataset analysis and visualization
├── Feature.ipynb            # Main model implementation
├── README.md               # This file
├── checkpoint/             # Model checkpoints
└── metadata.json          # Dataset metadata
```

## 🔧 Usage

### 1. Dataset Preparation
```python
# Load metadata
meta = "path/to/metadata.json"
df = pd.read_json(meta).T

# Split data
Train_set, Test_set = train_test_split(df, test_size=0.1, 
                                      random_state=42, stratify=df['label'])
```

### 2. Model Training
```python
# Prepare data
train_data, train_labels = prepare_all_videos(Train_set, "train")
test_data, test_labels = prepare_all_videos(Test_set, "test")

# Train model
history = model.fit([train_data[0], train_data[1]], train_labels,
                   validation_data=([test_data[0], test_data[1]], test_labels),
                   epochs=15, batch_size=8)
```

### 3. Prediction
```python
# Predict on new video
prediction = sequence_prediction("video_path.mp4")
if prediction >= 0.5:
    print("FAKE")
else:
    print("REAL")
```

## 🎯 Key Insights

### Dataset Characteristics
- **Imbalanced Classes**: 80.75% fake vs 19.25% real videos
- **Video Lengths**: Variable frame counts handled by sequence padding
- **Quality**: Videos processed to 224x224 resolution for consistency

### Model Performance
- **Stable Training**: Consistent accuracy across epochs
- **Good Generalization**: Similar training and validation performance
- **Efficient Processing**: Feature extraction reduces computational load

## 🔮 Future Improvements

1. **Data Augmentation**: Implement techniques to handle class imbalance
2. **Advanced Architectures**: Experiment with LSTM, Transformer models
3. **Multi-modal Fusion**: Combine audio and visual features
4. **Real-time Processing**: Optimize for live video streams
5. **Ensemble Methods**: Combine multiple model predictions

## 📚 References

- [TensorFlow Video Classification Tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
- [InceptionV3 Architecture](https://arxiv.org/abs/1512.00567)
- [GRU Networks](https://arxiv.org/abs/1412.3555)

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving model architecture
- Adding new features
- Optimizing performance
- Enhancing documentation

