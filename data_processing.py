# ---- Data Processing ---- #
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib


# Load CSV Data
df = pd.read_csv("hand_landmarks_Z.csv")

# Convert labels to numerical values (A → 0, B → 1 ect)
df["label"] = df["label"].map({"A": 0, "B": 1, "C": 2, "D":3, "E":4, "F" :5})

# Normalize landmark values (scaling between 0 and 1)
scaler = MinMaxScaler()
df[["x", "y", "z"]] = scaler.fit_transform(df[["x", "y", "z"]])
joblib.dump(scaler, "minmax_scaler.pkl")


# Reshape Data for TensorFlow
num_landmarks = 21
X = df[["x", "y", "z"]].values.reshape(-1, num_landmarks, 3)  # (num_frames, 21, 3)
y = df["label"].values  # Targets: 0 for A, 1 for B

print(df)


# Step 1: Group by frame and label, create input vectors
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
def preprocess_data(df):
    X = []
    y = []

    # Group by frame
    for frame, frame_group in df.groupby('frame'):
        # Within each frame, group by label (0 or 1)
        for label, label_group in frame_group.groupby('label'):
            # Ensure landmarks are sorted by landmark number (0 to 20)
            label_group = label_group.sort_values('landmark')
            # Verify that each label group has exactly 21 landmarks
            if len(label_group) == 21:
                # Extract x, y, z coordinates (21 landmarks * 3 coords = 63 values)
                features = label_group[['x', 'y', 'z']].values.flatten()
                X.append(features)
                y.append(label)
            else:
                print(f"Warning: Frame {frame}, Label {label} has {len(label_group)} landmarks instead of 21")

    # Convert to numpy arrays
    X = np.array(X)  # Shape: (n_samples, 63), where n_samples = 29 * 2 = 58
    y = np.array(y)  # Shape: (n_samples,)

    return X, y


# Preprocess the data
X, y = preprocess_data(df)

# Step 2: Normalize the features (optional but recommended)

from collections import Counter
print("Label distribution:", Counter(y))
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler to use during inference
joblib.dump(scaler, "scaler.pkl")

# Step 3: Split into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),  # Input layer: 63 features
    Dense(64, activation='relu'),                     # Hidden layer
    Dense(6, activation='softmax')                    # Output layer: binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test).argmax(axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Test Accuracy: {accuracy:.4f}")
model.save("hand_model.keras")