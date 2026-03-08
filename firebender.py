import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from utils import overlay_png

# --- SETUP ---
fire_img = cv2.imread('fireball.png', cv2.IMREAD_UNCHANGED)
fire_size = 200
fire_img = cv2.resize(fire_img, (fire_size, fire_size))

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

hand_states = {
    "Left":  {"speeds": deque([0]*5, maxlen=5), "fire_time": 0, "fire_pos": [0, 0], "fire_vel": [0.0, 0.0], "last_pos": [0, 0], "cooldown": 0},
    "Right": {"speeds": deque([0]*5, maxlen=5), "fire_time": 0, "fire_pos": [0, 0], "fire_vel": [0.0, 0.0], "last_pos": [0, 0], "cooldown": 0}
}

PUNCH_THRESHOLD = 35
FIST_RATIO = 0.18
FIRE_DURATION = 1.5
PROJECTILE_SPEED = 1.5 
COOLDOWN_TIME = 0.5

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    results = detector.detect_for_video(mp_image, timestamp)

    if results.hand_landmarks:
        for i, hand in enumerate(results.hand_landmarks):
            hand_label = results.handedness[i][0].category_name
            state = hand_states[hand_label]

            # landmarks
            wrist = hand[0]
            knuckle = hand[9]
            wx, wy = int(wrist.x * w), int(wrist.y * h)
            kx, ky = int(knuckle.x * w), int(knuckle.y * h)

            # speed
            if state["last_pos"] != [0, 0]:
                curr_speed = math.hypot(wx - state["last_pos"][0], wy - state["last_pos"][1])
                state["speeds"].append(curr_speed)
            avg_speed = sum(state["speeds"]) / len(state["speeds"])
            state["last_pos"] = [wx, wy]

            # fist check
            fist_dist = math.hypot(hand[12].x - wrist.x, hand[12].y - wrist.y)
            is_fist = fist_dist < FIST_RATIO

            # trigger logic
            current_time = time.time()
            if avg_speed > PUNCH_THRESHOLD and is_fist and (current_time - state["cooldown"] > COOLDOWN_TIME):
                dir_x, dir_y = kx - wx, ky - wy 
                mag = math.hypot(dir_x, dir_y)

                if mag != 0:
                    unit_x, unit_y = dir_x / mag, dir_y / mag
                    state["fire_time"] = current_time
                    state["cooldown"] = current_time

                    # spawn at the knuckles
                    state["fire_pos"] = [kx + int(unit_x * 50), ky + int(unit_y * 50)]
                    state["fire_vel"] = [unit_x * avg_speed * PROJECTILE_SPEED, 
                                         unit_y * avg_speed * PROJECTILE_SPEED]
                    
            indicator_color = (255,0,0) # blue
            if is_fist: 
                indicator_color = (0,255,0)
            if time.time() - state["fire_time"] < 0.2:
                indicator_color = (0,0,255) # red
            cv2.circle(frame, (wx, wy), 12, indicator_color, -1)

    # --- PHYSICS & DRAWING ---
    for label, state in hand_states.items():
        if time.time() - state["fire_time"] < FIRE_DURATION:
            # move fireball
            state["fire_pos"][0] += int(state["fire_vel"][0])
            state["fire_pos"][1] += int(state["fire_vel"][1])

            fx, fy = state["fire_pos"]
            tx, ty = fx - (fire_size // 2), fy - (fire_size // 2)

            frame = overlay_png(frame, fire_img, tx, ty)

    cv2.imshow("Bending Training", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Bending Training", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()