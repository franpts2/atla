import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from utils import overlay_png

fire_img = cv2.imread('fireball.png', cv2.IMREAD_UNCHANGED)

fire_size = 200
fire_img = cv2.resize(fire_img, (fire_size, fire_size))

# create hand landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

hand_states = {
    "Left":  {
        "speeds": deque([0]*5, maxlen=5), 
        "fire_time": 0, 
        "fire_pos": [0, 0],
        "fire_vel": [0.0, 0.0],
        "last_pos": [0, 0]
    },
    "Right": {
        "speeds": deque([0]*5, maxlen=5), 
        "fire_time": 0, 
        "fire_pos": [0, 0],  
        "fire_vel": [0.0, 0.0],
        "last_pos": [0, 0]   
    }
}
PUNCH_THRESHOLD = 35
FIST_RATIO = 0.18
FIRE_DURATION = 0.8 # seconds
PROJECTILE_SPEED = 2.0  # multiplier

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
            state = hand_states[hand_label]

            # track wrist
            wrist = hand[0]
            knuckle = hand[9]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            kx, ky = int(knuckle.x * w), int(knuckle.y * h)

            # direction vector from wrist to knuckle
            dx, dy = kx - wy, ky - wy

            # fire above the knuckle
            fire_spawn_x = wx + int(dx*2)
            fire_spawn_y = wy + int(dy*2)

            if state["last_pos"] != (0,0):
                curr_speed = math.hypot(wx - state["last_pos"][0], wy - state["last_pos"][1])
                state["speeds"].append(curr_speed)

            avg_speed = sum(state["speeds"]) / len(state["speeds"])
            state["last_pos"] = (wx, wy)

            # is hand closed?
            # comparing distance between middle finger tip (12) to wrist(0)
            finger_tip = hand[12]
            fist_dist = math.hypot(finger_tip.x - wrist.x, finger_tip.y - wrist.y)
            is_fist = fist_dist < FIST_RATIO

            
            if avg_speed > PUNCH_THRESHOLD and is_fist:
                dir_x = kx - wx
                dir_y = ky - wy

                mag = math.hypot(dir_x,dir_y)

                if mag != 0:
                    unit_x = dir_x / mag
                    unit_y = dir_y / mag

                    state["fire_time"] = time.time()
                    state["fire_pos"] = [fire_spawn_x, fire_spawn_y]
                    print(f"{hand_label} Fireball unleashed!")

                    state["fire_vel"] = [unit_x * avg_speed * PROJECTILE_SPEED,
                                         unit_y * avg_speed * PROJECTILE_SPEED]
                

            color = (0, 255, 0) if is_fist else (255, 0, 0)
            cv2.circle(frame, (wx, wy), 10, color, -1)

    # draw fire
    for label, state in hand_states.items():
        if time.time() - state["fire_time"] < FIRE_DURATION:
            state["fire_pos"][0] += int(state["fire_vel"][0])
            state["fire_pos"][1] += int(state["fire_vel"][1])

            fx, fy = state["fire_pos"]
            
            top_left_x = fx - (fire_size // 2)
            top_left_y = fy - (fire_size // 2)

            frame = overlay_png(frame, fire_img, top_left_x, top_left_y)

    cv2.imshow("Bending Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Q
        break
    if cv2.getWindowProperty("Bending Training", cv2.WND_PROP_VISIBLE) < 1: # window closed
        break

cap.release()
cv2.destroyAllWindows()
