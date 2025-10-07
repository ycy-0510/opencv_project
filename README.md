# OpenCV Project
This project demonstrates various computer vision techniques using OpenCV, including object detection, tracking, and image processing.

## Features
- Vehicle detection and tracking
- Face recognition
- Auto Document Scanner

# Installation
```bash
pip install -r requirements.txt
```

## How to Run
### Vehicle Tracking
1. Start the vehicle tracking API
```bash
python api.liveapi.py
```
2. Start the Streamlit app
```bash
streamlit run view/app.py
```
3. Open your browser and navigate to `http://localhost:8501`
### Face Recognition
- Blur faces in real-time using your webcam
```bash
python model/face/blur_face.py
```
- Recognize faces using your webcam
1. Add faces to the database (run this once per person)
```bash
python model/face/add_face.py
```
And input a 2 digit ID
2. Train the face recognition model (run this once after adding faces)
```bash
python model/face/train.py
```
3. Start the face recognition
```bash
python model/face/face_recognition.py
```
### Auto Document Scanner
```bash
python model/doc_scan/scan.py
```
