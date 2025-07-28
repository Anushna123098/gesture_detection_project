import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib

# === Step 1: Initialize ===
data_dir = "landmark_data"
X = []
y = []

# === Step 2: Load Data ===
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        label = filename.replace(".csv", "")

        try:
            # Read CSV without headers
            df = pd.read_csv(filepath, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')  # Ensure numeric
            df.dropna(inplace=True)  # Remove NaN rows

            # Skip if columns are too few (invalid landmark file)
            if df.shape[1] < 10:
                print(f"⚠️ File '{filename}' has too few columns. Skipping.")
                continue

            if not df.empty:
                X.extend(df.values)
                y.extend([label] * len(df))
                print(f"✅ Loaded {len(df)} samples for '{label}'")
            else:
                print(f"⚠️ Skipping empty file: {filename}")

        except Exception as e:
            print(f"❌ Error reading '{filename}': {e}")

# === Step 3: Convert to NumPy arrays ===
X = np.array(X, dtype=np.float32)
y = np.array(y)

if len(X) == 0 or len(y) == 0:
    print("❌ No data loaded. Check if CSV files contain valid numeric data.")
    exit()

# === Step 4: Encode Labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Save label encoder for use in real-time detection
joblib.dump(le, "label_encoder.pkl")
print("✅ Label encoder saved as 'label_encoder.pkl'")

# === Step 5: Split Dataset ===
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# === Step 6: Define Model Architecture ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 7: Save Best Model Automatically ===
checkpoint = ModelCheckpoint('gesture_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# === Step 8: Train Model ===
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[checkpoint])

print("✅ Training completed. Best model saved as 'gesture_model.h5'")
