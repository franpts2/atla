import math
import time
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

hands_history = {"Left": (0, 0), "Right": (0, 0)}
punch_threshold = 30

fire_duration = 0.5 # seconds
last_fire_time = 0
fire_pos = (0,0)

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
        for i, hand in enumerate(results.hand_landmarks):
            hand_label = results.handedness[i][0].category_name

            index_tip = hand[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            last_x, last_y = hands_history[hand_label]

            if last_x != 0:
                distance_moved = math.hypot(cx - last_x, cy - last_y)

                # check if it's a punch
                if distance_moved > punch_threshold:
                    last_fire_time = time.time() # curr time
                    fire_pos=(cx,cy)
                else:
                    cv2.circle(frame, (cx,cy), 15, (255,0,0), cv2.FILLED)

            hands_history[hand_label] = (cx,cy)

    if time.time() - last_fire_time < fire_duration:
        # Draw the lingering fire at the saved position
        cv2.circle(frame, fire_pos, 70, (0, 0, 255), -1)
        cv2.circle(frame, fire_pos, 85, (0, 165, 255), 5) # Orange outer ring
        cv2.putText(frame, "FIRE!", (fire_pos[0]-40, fire_pos[1]-100), 
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)


    cv2.circle(frame, (center_x, center_y), 50, (0,255,0),2)

    cv2.imshow("Bending Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Q
        break
    if cv2.getWindowProperty("Bending Training", cv2.WND_PROP_VISIBLE) < 1: # window closed
        break

cap.release()
cv2.destroyAllWindows()

