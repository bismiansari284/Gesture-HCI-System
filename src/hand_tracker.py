# src/hand_tracker.py

import cv2
import mediapipe as mp
import math

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gesture(self, frame):
        """
        Detects gestures: lock (fist), unlock (open palm)
        Returns:
            gesture (str) or None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            # Landmarks
            wrist = hand.landmark[0]
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]
            ring_tip = hand.landmark[16]
            pinky_tip = hand.landmark[20]

            # Helper function: distance between two landmarks
            def dist(a, b):
                return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

            # ======================
            # Detect lock (fist)
            # All fingertips close to wrist
            fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            if all(dist(f, wrist) < 0.1 for f in fingertips):
                return "lock"

            # ======================
            # Detect unlock (open palm)
            # All fingertips above wrist
            if all(f.y < wrist.y for f in fingertips):
                return "unlock"

            # ======================
            # Other gestures (stub)
            # Add more gestures here if needed
            # Example: index finger up → left click
            if index_tip.y < wrist.y and all(f.y > wrist.y for f in [middle_tip, ring_tip, pinky_tip]):
                return "left click"

            # Example: index + middle → right click
            if index_tip.y < wrist.y and middle_tip.y < wrist.y:
                return "right click"

        return None

    def release(self):
        self.hands.close()
