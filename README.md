# ATLA — Hand Tracking

A hand-tracking and gesture-recognition experiment inspired by Avatar: The Last Airbender.

<p align="left">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge" alt="MediaPipe">
</p>

## About

ATLA is a hand tracking and gesture recognition project using Python, OpenCV, and MediaPipe. 

Inspired by "Avatar: The Last Airbender," ATLA aims to be a bending simulator that recognizes and interprets gestures for all four elemental bending forms: fire, earth, water, and air. 

Currently, only fire bending is implemented, with plans to add the other elements in future updates. 

The project leverages state-of-the-art machine learning models to detect and analyze hand landmarks in real time, enabling gesture-based interactions and applications.

## Features

- Real-time hand detection and tracking
- Gesture recognition using MediaPipe's hand landmark model
- Bending simulator inspired by Avatar: The Last Airbender
- Currently supports fire bending gestures (more coming soon)
- Easy-to-use Python interface
- Modular utility functions for image processing
- Ready for integration with custom gesture logic

## How It Works

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

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install mediapipe opencv-python
   ```
2. **Download the model:**
   Download `hand_landmarker.task` from [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the project root.

## Roadmap

- Earth, water, and air bending gesture recognition
- Custom gesture recognition
- Visualization improvements
- Integration with other input modalities

## License

No license file is included. The source is public for reference; all rights reserved.

---

_ATLA — firebending, one gesture at a time._
