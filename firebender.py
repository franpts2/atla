import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# create hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

# hand's last position
prev_x, prev_y = 0, 0
punch_threshold = 50 # 50px needed to move

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    # convert OpenCV img to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # timestamp (required in video mode)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # detect hands
    results = detector.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        for hand in results.hand_landmarks:
            # index finger = landmark 8
            index_tip = hand[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            distance_moved = math.hypot(cx - prev_x, cy - prev_y)

            # check if it's a punch
            if distance_moved > punch_threshold:
                cv2.circle(frame, (cx, cy), 60, (0, 0, 255), -1) # fire circle
                cv2.putText(frame, "FIRE!", (cx-50, cy-80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
            else:
                cv2.circle(frame, (cx,cy), 15, (255,0,0), cv2.FILLED)

            prev_x, prev_y = cx, cy


    cv2.circle(frame, (center_x, center_y), 50, (0,255,0),2)

    cv2.imshow("Bending Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Q
        break
    if cv2.getWindowProperty("Bending Training", cv2.WND_PROP_VISIBLE) < 1: # window closed
        break

cap.release()
cv2.destroyAllWindows()

