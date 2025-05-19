# ---- Data Processing ---- #
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


# Load CSV Data
df = pd.read_csv("hand_landmarks_A.csv")

# Convert labels to numerical values (A → 0, B → 1)
df["label"] = df["label"].map({"A": 0, "B": 1})

# Normalize landmark values (scaling between 0 and 1)
scaler = MinMaxScaler()
df[["x", "y", "z"]] = scaler.fit_transform(df[["x", "y", "z"]])

# Reshape Data for TensorFlow
num_landmarks = 21
X = df[["x", "y", "z"]].values.reshape(-1, num_landmarks, 3)  # (num_frames, 21, 3)
y = df["label"].values  # Targets: 0 for A, 1 for B

print(df)
