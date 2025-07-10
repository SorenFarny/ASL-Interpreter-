import mediapipe as mp
import cv2
import csv



# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Open CSV file for writing hand landmark data
with open("hand_landmarks_Z.csv", mode="a", newline="") as file:
    writer = csv.writer(file)

    # Write header row
    writer.writerow(["frame", "landmark", "x", "y", "z", "label"])  # Added 'label' column

    i = 30  # Frame counter
    j=666 #index counter
    # Loop to capture frames
    while True:
        success, image = cap.read()
        if not success:
            print("Failed to capture image")
            break

        # Convert image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(image_rgb)

        # Draw hand landmarks and save data every 15 frames
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if i >= 15:
                    i = 0
                    j+=1
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        writer.writerow([j, idx, landmark.x, landmark.y, landmark.z, "K"
                                                                                     ""])  # Labeling the row

        i += 1  # Increment frame counter

        # Display the image
        cv2.imshow("Hand Tracking", image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()