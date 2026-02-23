import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import pyautogui
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from system_control import SystemControl
from hand_tracker import HandTracker
import time
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ==============================
# Load Model
# ==============================

try:
    model = joblib.load("../model/gesture_model.pkl")
    print("Model Loaded Successfully")

except Exception as e:
    print("Model Error:", e)
    exit()

# ==============================
# System Control
# ==============================

system_control = SystemControl()

# ==============================
# Buffers
# ==============================

gesture_history = deque(maxlen=10)
lock_history = deque(maxlen=15)

# ==============================
# Variables
# ==============================

last_action_time = 0
ACTION_DELAY = 0.8

prev_time = 0
fps_buffer = deque(maxlen=10)

prev_x = 0
prev_y = 0
smoothening = 12
current_mode = "Cursor"

screen_w, screen_h = pyautogui.size()

cam_w = 640
cam_h = 480

stable_prediction = "None"
confidence = 0

# ==============================
# Auto Calibration
# ==============================

calibration_frames = 50
calibration_data = []

calibrated = False

auto_smoothening = smoothening

# ==============================
# Volume Setup
# ==============================

try:

    devices = AudioUtilities.GetSpeakers()

    interface = devices.Activate(
        IAudioEndpointVolume._iid_,
        CLSCTX_ALL,
        None
    )

    volume = cast(interface,
                  POINTER(IAudioEndpointVolume))

    volume_control = True

except:

    print("Volume Control Disabled")
    volume_control = False

# ==============================
# MediaPipe
# ==============================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

tracker = HandTracker()

# ==============================
# Camera
# ==============================

cap = cv2.VideoCapture(0)

cap.set(3, cam_w)
cap.set(4, cam_h)

if not cap.isOpened():
    print("Camera Not Found")
    exit()

print("Gesture System Started")

# ==============================
# Main Loop
# ==============================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    status_text = "ACTIVE"

    # ======================
    # Hand Detection
    # ======================

    if results.multi_hand_landmarks:
        

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        wrist = hand.landmark[0]

        landmarks = []

        for lm in hand.landmark:

            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y
            ])

        prediction = model.predict(
            [landmarks])[0]

        probabilities = model.predict_proba(
            [landmarks])[0]

        confidence = max(probabilities)

        gesture_history.append(prediction)

        stable_prediction = max(
            set(gesture_history),
            key=gesture_history.count)
        # ======================
        # Mode Switching
        # ======================
        if stable_prediction == "Cursor":
            current_mode = "Cursor"
        elif stable_prediction == "scroll":
            current_mode = "Scroll"
        elif stable_prediction == "Brightness":
            current_mode = "Brightness"
        elif stable_prediction == "Volume":
            current_mode = "Volume"

        # Fingers

        thumb = hand.landmark[4]
        index = hand.landmark[8]
        # ======================
        # Auto Calibration
        # ======================
        if not calibrated:
            calibration_data.append(
                (index.x, index.y)
        )
            cv2.putText(frame,
            "Calibrating...",
            (200,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            3)
            if len(calibration_data) > calibration_frames:
                xs = [p[0] for p in calibration_data]
                ys = [p[1] for p in calibration_data]
                std_x = np.std(xs)
                std_y = np.std(ys)
                shake = (std_x + std_y)/2
                auto_smoothening = int(
                    8 + shake*50
                )
                calibrated = True
                print("Calibration Complete")
                print("Smoothening =",auto_smoothening)
        middle = hand.landmark[12]
        ring = hand.landmark[16]
        pinky = hand.landmark[20]

        # ======================
        # Cursor Control
        # ======================

        if current_mode == "Cursor":

            x = index.x * cam_w
            y = index.y * cam_h

            screen_x = np.interp(
                x,
                (0, cam_w),
                (0, screen_w)
            )

            screen_y = np.interp(
                y,
                (0, cam_h),
                (0, screen_h)
            )

            curr_x = prev_x + (
                screen_x - prev_x
            ) / auto_smoothening

            curr_y = prev_y + (
                screen_y - prev_y
            ) / auto_smoothening

            pyautogui.moveTo(
                curr_x,
                curr_y)

            prev_x = curr_x
            prev_y = curr_y

    # ======================
    # FPS
    # ======================

    current = time.time()

    fps = 1/(current-prev_time) if prev_time!=0 else 0

    prev_time = current

    fps_buffer.append(fps)

    avg_fps = int(
        sum(fps_buffer)/len(fps_buffer)
    )

    # ======================
    # Professional UI
    # ======================

    cv2.rectangle(
        frame,
        (0,0),
        (300,200),
        (0,0,0),
        -1
    )

    cv2.putText(frame,
    "Gesture HCI System",
    (10,20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,255),
    2)

    cv2.putText(frame,
    f"Status: {status_text}",
    (10,50),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0,255,255),
    2)

    cv2.putText(frame,
    f"Gesture: {stable_prediction}",
    (10,80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0,255,0),
    2)

    cv2.putText(frame,
    f"Confidence: {int(confidence*100)}%",
    (10,110),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0,255,0),
    2)

    cv2.putText(frame,
    f"FPS: {avg_fps}",
    (10,140),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,0),
    2)

    cv2.putText(frame,
    f"Mode: {current_mode}",
    (10,170),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,0,255),
     2)
    

    cv2.putText(frame,
    f"Smooth: {auto_smoothening}",
    (10,200),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255,255,255),
    2)

    cv2.imshow(
        "Gesture HCI",
        frame
    )

    key=cv2.waitKey(1)

    if key==27 or key==ord('q'):
        break

cap.release()

cv2.destroyAllWindows()