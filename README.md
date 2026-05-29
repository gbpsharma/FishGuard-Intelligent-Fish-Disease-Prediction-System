# FishGuard – Intelligent Fish Disease Prediction System

FishGuard is an AI-powered fish disease diagnosis platform that automatically detects fish diseases from uploaded images using Deep Learning and Computer Vision techniques. The system validates fish images, detects fish regions using YOLOv8, classifies diseases using a custom ResNet50-based architecture, and visualizes infected areas through Grad-CAM explainable AI.

The project provides an end-to-end solution consisting of a ReactJS frontend, FastAPI backend, YOLO object detection, custom deep learning classifier, and Grad-CAM visualization.

---

## Features

- Fish Image Validation using ImageNet-pretrained ResNet18
- Fish Detection and Cropping using YOLOv8
- Fish Disease Classification using Custom Deep Learning Architecture
- Channel Attention Mechanism for Improved Feature Learning
- OSELM-based Classification Layer
- Grad-CAM Explainable AI Visualization
- Infected Area Localization
- Healthy Fish Identification
- Non-Fish Image Rejection
- Top-3 Disease Prediction Results
- Confidence Score Estimation
- Modern ReactJS User Interface
- FastAPI REST API Backend

---

## Problem Statement

Fish diseases significantly impact aquaculture productivity and profitability. Manual diagnosis requires expert knowledge and can be time-consuming.

FishGuard addresses this challenge by providing an automated disease detection system capable of:

- Identifying common fish diseases
- Highlighting disease-affected regions
- Providing confidence-based predictions
- Assisting aquaculture professionals with faster diagnosis

---

## System Workflow

```text
Input Fish Image
        │
        ▼
Fish Validation
(ResNet18)
        │
        ▼
Fish Detection
(YOLOv8)
        │
        ▼
Fish Cropping
        │
        ▼
Disease Classification
(Custom ResNet50 + Attention + OSELM)
        │
        ▼
Grad-CAM Generation
        │
        ▼
Infected Area Localization
        │
        ▼
Prediction Results
```

---

## Model Architecture

### Feature Extraction

The feature extraction module is based on:

- ResNet50 Backbone
- Multi-Layer Feature Fusion
- Channel Attention Mechanism
- Adaptive Average Pooling

### Classification Module

The classifier uses:

- OSELM (Online Sequential Extreme Learning Machine)
- Hidden Layer Size: 1024
- Output Layer: 7 Classes

### Architecture Pipeline

```text
Input Image
     │
     ▼
ResNet50 Backbone
     │
     ├── Layer2 Features
     ├── Layer3 Features
     └── Layer4 Features
              │
              ▼
      Channel Attention
              │
              ▼
Feature Fusion
              │
              ▼
Adaptive Pooling
              │
              ▼
OSELM Classifier
              │
              ▼
Disease Prediction
```

---

## Project Structure

```text
MAJORPROJECT
│
├── backend
│   ├── main.py
│   └── yolov8n.pt
│
├── frontend
│   ├── public
│   │   └── index.html
│   │
│   ├── src
│   │   ├── components
│   │   │   ├── controlpanel.js
│   │   │   ├── resultpanel.js
│   │   │   └── uploadpanel.js
│   │   │
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   │
│   ├── package.json
│   └── package-lock.json
│
├── data
│   └── fishdiseasedataset
│       ├── Train
│       └── Test
│
├── model_definition.py
├── train.py
├── predict.py
├── fish_disease_model.pth
├── README.md
└── .gitignore
```

---

## Disease Classes

The model supports classification of the following disease categories:

1. Bacterial Red Disease
2. Bacterial Diseases – Aeromoniasis
3. Bacterial Gill Disease
4. Fungal Disease – Saprolegniasis
5. Healthy Fish
6. Parasitic Diseases
7. Viral Disease – White Tail Disease

---

## Technologies Used

### Frontend

- ReactJS
- Axios
- HTML5
- CSS3

### Backend

- FastAPI
- Python

### Deep Learning

- PyTorch
- Torchvision
- TIMM

### Computer Vision

- OpenCV
- Pillow (PIL)
- NumPy

### Object Detection

- YOLOv8 (Ultralytics)

### Explainable AI

- Grad-CAM

---

## Dataset

The model is trained on a fish disease dataset organized as:

```text
fishdiseasedataset
│
├── Train
│   ├── Bacterial Red disease
│   ├── Bacterial diseases - Aeromoniasis
│   ├── Bacterial gill disease
│   ├── Fungal diseases Saprolegniasis
│   ├── Healthy Fish
│   ├── Parasitic diseases
│   └── Viral diseases White tail disease
│
└── Test
    ├── Bacterial Red disease
    ├── Bacterial diseases - Aeromoniasis
    ├── Bacterial gill disease
    ├── Fungal diseases Saprolegniasis
    ├── Healthy Fish
    ├── Parasitic diseases
    └── Viral diseases White tail disease
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/FishGuard.git
cd FishGuard
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

---

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / MacOS

```bash
source venv/bin/activate
```

---

### Install Python Dependencies

```bash
pip install fastapi
pip install uvicorn
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install numpy
pip install timm
pip install ultralytics
pip install python-multipart
pip install scikit-learn
pip install matplotlib
```

---

## Running the Backend

Navigate to backend directory:

```bash
cd backend
```

Start FastAPI server:

```bash
uvicorn main:app --reload
```

Backend API:

```text
http://127.0.0.1:8000
```

---

## Running the Frontend

Navigate to frontend directory:

```bash
cd frontend
```

Install packages:

```bash
npm install
```

Start React application:

```bash
npm start
```

Frontend URL:

```text
http://localhost:3000
```

---

## 🔌 API Endpoint

### Predict Fish Disease

#### Request

```http
POST /predict
```

#### Input

Multipart Image File

#### Response

```json
{
	"predicted_class":"Healthy Fish",
	"confidence":98.52,
	"image":"base64_string",
	"gradcam":"base64_string",
	"infected":"base64_string",
	"top3":[
		["Healthy Fish",98.52],
		["Parasitic diseases",0.95],
		["Bacterial gill disease",0.53]
	]
}
```

---

## Grad-CAM Visualization

The system generates explainable AI outputs:

### Original Image

Displays the uploaded fish image.

### Grad-CAM Heatmap

Highlights image regions contributing to disease prediction.

### Infected Area Detection

Draws a bounding box around the most probable infected region.

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Backbone | ResNet50 |
| Attention | Channel Attention |
| Classifier | OSELM |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Epochs | 20 |
| Image Size | 224 × 224 |

---

## Model Validation Pipeline

Before disease classification, FishGuard performs:

1. Fish Image Validation using ResNet18
2. Fish Detection using YOLOv8
3. Fish Cropping
4. Disease Classification

This prevents incorrect predictions on non-fish images.

---

## Key Contributions

- Developed custom fish disease classification architecture.
- Integrated YOLOv8 for fish localization.
- Implemented Grad-CAM for explainable predictions.
- Added infected region highlighting.
- Built complete ReactJS + FastAPI deployment pipeline.
- Designed an automated fish disease diagnosis workflow.

---

## Future Enhancements

- Real-Time Webcam Detection
- Mobile Application Deployment
- Cloud-Based Prediction Service
- Multi-Fish Detection Support
- Disease Severity Estimation
- Treatment Recommendation System
- Aquaculture Analytics Dashboard

---

## ⚠️ Important Note

The following files may not be included in the GitHub repository due to their large size:

```text
fish_disease_model.pth
backend/yolov8n.pt
data/fishdiseasedataset
```

To run the project successfully:

1. Place `fish_disease_model.pth` in the project root directory.
2. Place `yolov8n.pt` inside the backend folder.
3. Download and place the dataset in:

```text
data/fishdiseasedataset
```

---

## Author

### Bhanu Prakash Sharma

B.Tech – Computer Science and Engineering  
Artificial Intelligence & Machine Learning

---

This project is developed for academic, educational and research purposes.