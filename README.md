# ATLA - Hand Tracking Project

<div align="left">
	<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
	<img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
	<img src="https://img.shields.io/badge/MediaPipe-FF6F00?style=for-the-badge&logo=google&logoColor=white" alt="MediaPipe"/>
</div>

## About

ATLA is a hand tracking and gesture recognition project using Python, OpenCV, and MediaPipe. 

Inspired by "Avatar: The Last Airbender," ATLA aims to be a bending simulator that recognizes and interprets gestures for all four elemental bending forms: fire, earth, water, and air. 

Currently, only fire bending is implemented, with plans to add the other elements in future updates. 

The project leverages state-of-the-art machine learning models to detect and analyze hand landmarks in real time, enabling gesture-based interactions and applications.

## Key Features

- Real-time hand detection and tracking
- Gesture recognition using MediaPipe's hand landmark model
- Bending simulator inspired by Avatar: The Last Airbender
- Currently supports fire bending gestures (more coming soon)
- Easy-to-use Python interface
- Modular utility functions for image processing
- Ready for integration with custom gesture logic

## Demo

_Coming soon!_

## Technical Implementation

**Tech Stack**

- Language: Python 3.8+
- Computer Vision: OpenCV
- ML Model: MediaPipe Hand Landmarker

**System Architecture**

```
┌──────────────────────────────┐
│        Python Script         │
│  OpenCV + MediaPipe Model    │
├──────────────────────────────┤
│   Real-time Video Capture    │
└──────────────────────────────┘
```

**Folder Structure**

```
atla/
├── firebender.py           # Main script for hand tracking
├── utils.py                # Utility functions
├── hand_landmarker.task    # MediaPipe hand model
├── README.md               # Project documentation
└── __pycache__/            # Python cache files
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install mediapipe opencv-python
   ```
2. **Download the model:**
   Download `hand_landmarker.task` from [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the project root.

## Features Implemented

- Hand detection and landmark extraction
- Real-time video processing with OpenCV
- Modular code for easy extension

## Planned Features

- Earth, water, and air bending gesture recognition
- Custom gesture recognition
- Visualization improvements
- Integration with other input modalities

---

_ATLA - Hand Tracking Project by Francisca Portugal_
