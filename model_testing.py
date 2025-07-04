import cv2
import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model
import mediapipe as mp

# Load model and scalers
model = load_model("hand_model.keras")
minmax_scaler = load("minmax_scaler.pkl")
standard_scaler = load("scaler.pkl")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

print("🤖 Ready to recognize hand gestures. Press 'q' to quit.")

while True:
    success, image = cap.read()
    if not success:
        print("🚫 Failed to capture image")
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    landmark_vector = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx in range(21):
                lm = hand_landmarks.landmark[idx]
                landmark_vector.extend([lm.x, lm.y, lm.z])
            break  # Only process the first detected hand

    if len(landmark_vector) != 63:
        print("⚠️ Incomplete or missing landmarks — skipping frame")
        continue

    # Step 1: reshape and add feature names for MinMaxScaler
    landmarks_np = np.array(landmark_vector).reshape(21, 3)
    landmarks_df = pd.DataFrame(landmarks_np, columns=["x", "y", "z"])

    # Step 2: scale with MinMaxScaler
    landmarks_minmax = minmax_scaler.transform(landmarks_df)

    # Step 3: flatten and scale with StandardScaler
    flat_input = landmarks_minmax.flatten().reshape(1, -1)
    final_input = standard_scaler.transform(flat_input)

    # Step 4: prediction
    prediction = model.predict(final_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    gesture = label_map[predicted_class]

    print("🧠 Probabilities:", prediction)
    print("🖐️ Predicted gesture:", gesture)

    # Show camera feed
    cv2.imshow("Hand Tracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()