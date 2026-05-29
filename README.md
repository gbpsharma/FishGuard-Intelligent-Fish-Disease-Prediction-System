# FishGuard – Intelligent Fish Disease Prediction System

FishGuard is an AI-powered fish disease detection and diagnosis system developed using Deep Learning, Computer Vision, FastAPI, ReactJS, YOLOv8 and Grad-CAM visualization.

The system automatically detects fish from uploaded images, classifies fish diseases, highlights infected regions using Grad-CAM, and provides confidence scores with top predictions.

---

## Features

- Fish Image Validation using ImageNet-pretrained ResNet18
- Fish Detection using YOLOv8
- Fish Disease Classification using Custom ResNet50 + Channel Attention + OSELM Architecture
- Grad-CAM Disease Visualization
- Infected Area Localization
- Top-3 Disease Predictions
- Healthy Fish Detection
- Non-Fish Image Rejection
- ReactJS Interactive Frontend
- FastAPI Backend API

---

## System Architecture

Input Fish Image
↓
Fish Validation (ResNet18)
↓
Fish Detection (YOLOv8)
↓
Fish Cropping
↓
Disease Classification Model
↓
Grad-CAM Generation
↓
Infected Area Detection
↓
Prediction Results

---

## Technologies Used

### Frontend
- ReactJS
- Axios
- CSS3

### Backend
- FastAPI
- Python

### Deep Learning
- PyTorch
- Torchvision
- TIMM
- YOLOv8 (Ultralytics)

### Computer Vision
- OpenCV
- PIL

### Explainable AI
- Grad-CAM

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
│   ├── src
│   │   ├── components
│   │   ├── App.js
│   │   ├── App.css
│   │   └── index.js
│   │
│   ├── package.json
│   └── README.md
│
├── model_definition.py
├── train.py
├── predict.py
├── fish_disease_model.pth
├── data/
└── README.md
```

---

## Custom Model Architecture

### Feature Extraction

- ResNet50 Backbone
- Multi-layer Feature Fusion
- Channel Attention Mechanism

### Classification

- OSELM Classifier
- 1024 Hidden Neurons
- 7 Fish Disease Classes

---

## Supported Classes

1. Bacterial Red Disease
2. Bacterial Diseases – Aeromoniasis
3. Bacterial Gill Disease
4. Fungal Disease – Saprolegniasis
5. Healthy Fish
6. Parasitic Diseases
7. Viral Disease – White Tail Disease

---

## Dataset

Dataset contains fish images categorized into seven disease classes.

Directory Structure:

```text
data/
└── fishdiseasedataset/
    ├── Train/
    └── Test/
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/FishGuard.git
cd FishGuard
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running Backend

Move to backend folder:

```bash
cd backend
```

Run FastAPI:

```bash
uvicorn main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Running Frontend

Move to frontend folder:

```bash
cd frontend
```

Install packages:

```bash
npm install
```

Start React Application:

```bash
npm start
```

Frontend URL:

```text
http://localhost:3000
```

---

## API Endpoint

### POST

```http
POST /predict
```

### Input

Multipart Image File

### Output

```json
{
    "predicted_class":"Healthy Fish",
    "confidence":98.52,
    "image":"base64",
    "gradcam":"base64",
    "infected":"base64",
    "top3":[]
}
```

---

## Explainable AI

Grad-CAM is used to visualize the important disease regions responsible for prediction.

The system generates:

- Original Fish Image
- Grad-CAM Heatmap
- Infected Region Bounding Box

---

## Future Enhancements

- Real-Time Camera Detection
- Mobile Application
- Multi-Fish Detection
- Disease Treatment Recommendation
- Cloud Deployment
- Disease Severity Analysis

---

## Author

**Bhanu Prakash Sharma**

B.Tech – Computer Science and Engineering

Artificial Intelligence & Machine Learning

---

This project is developed for academic and research purposes.