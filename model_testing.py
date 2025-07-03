import mediapipe as mp
import cv2
import csv
import pandas as pd



# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()


while True:
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(image_rgb)

        import numpy as np

        landmark_vector = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Sort and collect all 21 landmarks
                for idx in range(21):  # Force only first 21 landmarks
                    lm = hand_landmarks.landmark[idx]
                    landmark_vector.extend([lm.x, lm.y, lm.z])  # Flat list: 63 values

                break  # Use only the first detected hand
        print(landmark_vector)

        cv2.imshow("Hand Tracking", image)
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #print(df)
            break

# Release resources
cap.release()
cv2.destroyAllWindows()