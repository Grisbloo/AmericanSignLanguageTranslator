# Real-Time American Sign Language Translator (V2 Desktop)

![ASL Translator Demo](AmericanSignLanguage/asl_translateV2/asssets/ASLTranslator.gif)

*Model Performance — Epoch: 49, Train Loss: 0.03*

## Overview
This project is a real-time American Sign Language (ASL) interpreter that converts hand gestures into text. It uses a multi-stage machine learning pipeline optimized for desktop environments:

1. **Object Detection:** A custom-trained YOLO model scans the webcam feed to isolate the user’s hand and crop out background noise.
2. **Feature Extraction:** MediaPipe Hands processes the cropped frame into a 21-point skeletal representation, normalized to the wrist.
3. **Temporal Classification:** A PyTorch Long Short-Term Memory (LSTM) network utilizes a 16-frame sliding window to analyze hand motion and classify the ASL letter (including dynamic gestures like J and Z).

The output is projected via a custom OpenCV state-machine UI for robust, real-time translation.

## Installation & Setup

**Requirements:** Python 3.10+, a webcam, and a Windows/Mac/Linux desktop.

**1. Setup Virtual Environment**
```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**2. Install Dependencies**
```bash
pip install opencv-python mediapipe ultralytics torch torchvision numpy
```

**3. Add Neural Network Weights**
Place the following three files directly into the root directory of the project:
* `best.pt` (Custom YOLO hand-tracking weights)
* `hand_landmarker.task` (Official MediaPipe weights)
* `Bestest_model.pth` (Custom trained LSTM weights)

## Usage
Run the live inference pipeline:
```bash
python main.py
```
*(Press 'q' while the webcam window is active to safely shut down the pipeline and release the camera).*
