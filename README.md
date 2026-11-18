# Real-Time Face Recognition System

A local Windows desktop application that detects and recognizes faces from a webcam using OpenCV YuNet and SFace. Camera frames and face data are processed and stored on this computer; they are not sent to a service.

Register people only with their knowledge and permission. Registration stores face images and numeric face embeddings locally. The embeddings are biometric data, so protect the project folder and remove registrations you no longer need.

## Features

- Live webcam preview with multiple-face detection and labeled bounding boxes.
- Local face registration with a name and three captured face samples.
- SFace embeddings stored with registration metadata in `dataset/registered_faces/registry.json`; captured images are stored in that person's local subfolder.
- Unknown-face labeling and cosine similarity display. Similarity is a model score, **not** a probability or proof of identity.
- Tkinter interface and basic tests for registration, storage, loading, matching, and unknown faces.

## Technology

- Python 3.11+
- OpenCV contrib (YuNet detector and SFace recognizer)
- NumPy, Pillow, Tkinter, and JSON

The dlib-based `face-recognition` package is not used. YuNet and SFace are used together through OpenCV's face APIs.

## Project structure

```text
.
├── dataset/
│   └── registered_faces/
├── models/
├── src/
│   ├── camera.py
│   ├── database.py
│   ├── face_detection.py
│   ├── face_recognition.py
│   ├── face_registration.py
│   ├── main.py
│   └── utils.py
├── tests/
├── assets/
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

Python 3.11 must be installed. Run these commands from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The application requires the two OpenCV Zoo ONNX model files in `models`. Download the official YuNet and SFace models:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o "models\face_detection_yunet_2023mar.onnx"
curl.exe -L "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" -o "models\face_recognition_sface_2021dec.onnx"
```

The downloads are used only as local model files. Once present, camera processing and recognition run locally. If downloading in a browser, save the files under the exact names shown above. The app checks for missing model files and reports what is needed.

## Run

```powershell
.\.venv\Scripts\python.exe -m src.main
```

Click **Start Camera** to begin live detection. **Stop Camera** releases the webcam; **Exit** also stops it and closes the application.

## Register a face

1. Ensure the person has agreed to be registered.
2. Click **Register Face**. The app starts the camera if needed.
3. Enter the person's name.
4. With only that person in view, capture three clear samples. Slightly changing the angle between samples can help.
5. Click **Save Registration**. Images and embeddings are saved locally under `dataset/registered_faces`.

Names must not be blank. Registration is not saved until all three samples have been captured. To remove an identity, close the app and remove its identity record from `registry.json` and the matching image subfolder.

## Test recognition

Run the app, register a consenting person, then view them in the camera. The name is shown when the cosine similarity meets the SFace match threshold; otherwise the face is labeled **UNKNOWN**. For multiple faces, each box is matched independently. The displayed similarity ranges from -1 to 1 and is not a calibrated confidence percentage.

Run the tests without a camera or model files:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **Models unavailable:** confirm both ONNX files exist in `models` and have the exact filenames above. Re-download any incomplete file.
- **Camera cannot open:** check Windows camera privacy permissions, connect the webcam, and close other apps using it.
- **No face detected:** improve lighting, face the camera, and move closer.
- **Registration data cannot load:** the app reports invalid JSON rather than overwriting it. Back up `registry.json` before making manual repairs.
- **Import or DLL errors:** activate/use this project's `.venv` and reinstall from `requirements.txt`; do not install packages globally.
