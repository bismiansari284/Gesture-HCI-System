import cv2
import mediapipe as mp
import csv
import os

gesture_name = input("Enter gesture label: ")
num_samples = 600

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

os.makedirs("dataset", exist_ok=True)
file_path = f"dataset/{gesture_name}.csv"

count = 0

with open(file_path, mode='w', newline='') as f:
    writer = csv.writer(f)

    print("Collecting samples... Hold gesture steady.")

    while count < num_samples:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            wrist = hand.landmark[0]
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y
                ])

            writer.writerow(landmarks)
            count += 1
            print(f"Collected: {count}")

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Done!")

cap.release()
cv2.destroyAllWindows()
