import mediapipe as mp
import cv2
import csv
import pandas as pd
from keras.models import load_model
import numpy as np

# Load model
model = load_model("hand_model.keras")



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
        if len(landmark_vector) != 63:
            print("Incomplete or missing landmarks — skipping frame")
            continue  # Skip this frame

        from joblib import load
        # Load scalers
        minmax_scaler = load("minmax_scaler.pkl")
        standard_scaler = load("scaler.pkl")

        # Step 1: shape (21, 3)
        landmarks_np = np.array(landmark_vector).reshape(21, 3)

        # Step 2: MinMax scaling
        landmarks_minmax = minmax_scaler.transform(landmarks_np)

        # Step 3: flatten to shape (1, 63)
        input_array = landmarks_minmax.flatten().reshape(1, -1)

        # Step 4: StandardScaler transform
        input_array = standard_scaler.transform(input_array)


        scaler = load("scaler.pkl")
        input_array = scaler.transform(input_array)

        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        #print(input_array)
        print("Probabilities:", model.predict(input_array))
        print("Predicted gesture:", label_map[predicted_class])

        cv2.imshow("Hand Tracking", image)
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #print(df)
            break

# Release resources
cap.release()
cv2.destroyAllWindows()