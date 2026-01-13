//Face Recognition using Deep Metric Learning
Overview

This project implements an end-to-end face recognition pipeline based on deep metric learning.
It combines YOLOv5 for face detection and embedding-based face recognition to identify individuals by comparing feature vectors against a database.

The system supports:

Building a face embedding database from images

Face recognition from videos

Real-time face recognition via API (webcam / mobile camera)

Pipeline

1 Face detection using YOLOv5

2 Face alignment & preprocessing

3 Embedding extraction using a face recognition model (ArcFace / FaceNet)

4 Similarity comparison (cosine similarity)

5 Identity matching from an embedding database

Installation
git clone git@github.com:tuanhleee/Face-Recognition-using-Deep-Metric-Learning.git
cd Face-Recognition-using-Deep-Metric-Learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Usage
1️⃣ Build the face embedding database

Use the notebook multiple_model.ipynb to convert images into face embeddings and store them in a database.

Steps:

Place face images inside the dataset folder (one folder per identity)

Open the notebook:

jupyter notebook multiple_model.ipynb


Run all cells to:

Detect faces

Extract embeddings

Save embeddings to the database (CSV / SQLite)

This step is required before any face recognition test.

2️⃣ Face recognition testing
Option A — Test with video file

Use multiple_model.py to perform face recognition on a video.

python multiple_model.py --video path/to/video.mp4


The script will:

Detect faces in each frame

Compare embeddings with the database

Display predicted identities in real time

Option B — Real-time testing via API (webcam / mobile)
Step 1: Start the API server

Run the FastAPI server using Uvicorn:

uvicorn api:app --host 0.0.0.0 --port 8000

Step 2: Expose the API using Ngrok

In another terminal:

ngrok http 8000


Ngrok will generate a public URL such as:

https://xxxx-xx-xx-xx.ngrok-free.app

Step 3: Test with a device camera

Open the Ngrok URL on a machine or mobile device with a camera

Grant camera access

Perform face recognition in real time directly from the browser

This allows cross-device testing (phone → laptop server).

Notes

The embedding database must be created before testing.

Large files (datasets, model weights, virtual environments) are ignored via .gitignore.

GPU acceleration is recommended for real-time performance.

Technologies

Python

YOLOv5

PyTorch

TensorFlow / Keras (for embeddings)

FastAPI

Uvicorn

Ngrok

OpenCV

Author
Tu Anh Le
Master  – Data Science & Artificial Intelligence
Université de Rouen Normandie
