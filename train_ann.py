import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# -------------------------
# Step 1: Load Dataset
# -------------------------
dataset_path = "dataset"

data = []
labels = []

IMG_SIZE = 64

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # 🔥 Skip files, only process folders
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path)

        # Skip invalid images
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(img)
        labels.append(person_name)

# -------------------------
# Step 2: Convert to NumPy
# -------------------------
data = np.array(data).reshape(len(data), IMG_SIZE, IMG_SIZE)
data = data / 255.0

# -------------------------
# Step 3: Encode Labels
# -------------------------
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# -------------------------
# Step 4: Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -------------------------
# Step 5: Build ANN Model
# -------------------------
model = Sequential()

model.add(Flatten(input_shape=(IMG_SIZE, IMG_SIZE)))

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# Step 6: Train Model
# -------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    validation_data=(X_test, y_test)
)

# -------------------------
# Step 7: Accuracy Check
# -------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\n=======================")
print("ANN MODEL RESULTS")
print("=======================")
print("Accuracy:", accuracy * 100, "%")

# -------------------------
# Step 8: Save Model
# -------------------------
model.save("models/ann_model.h5")

print("\nModel saved successfully!")

# -------------------------
# Step 9: Plot Accuracy Graph
# -------------------------
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("ANN Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()