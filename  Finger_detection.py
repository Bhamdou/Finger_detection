import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

tips_ids = [4, 8, 12, 16, 20]  # IDs of the fingertips in the mediapipe hand landmarks model

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

                x_list = []
                y_list = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)

                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)

                bbox = xmin, ymin, xmax, ymax

                fingers = []
                for id in tips_ids:
                    if handLms.landmark[id].y < handLms.landmark[id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Identify if this is a left hand or a right hand
                if handLms.classification[0].label == 'Right':
                    if fingers[0] == 1:
                        fingers[0] = 0
                    else:
                        fingers[0] = 1

                total_fingers = fingers.count(1)
                cv2.putText(img, f'Fingers:{total_fingers}', (bbox[0], bbox[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20),
                              (0, 255, 0), 2)

        cv2.imshow('Image', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
