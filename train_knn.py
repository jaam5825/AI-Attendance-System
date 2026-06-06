import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# Step 1: Load Dataset
# -------------------------
dataset_path = "dataset"

data = []
labels = []

IMG_SIZE = 64

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Only folders
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        data.append(img)
        labels.append(person_name)

# -------------------------
# Step 2: Convert to NumPy
# -------------------------
data = np.array(data)
data = data.reshape(len(data), IMG_SIZE * IMG_SIZE)
data = data / 255.0

# -------------------------
# Step 3: Encode Labels
# -------------------------
le = LabelEncoder()
labels = le.fit_transform(labels)

# -------------------------
# Step 4: Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -------------------------
# Step 5: Train KNN Model
# -------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -------------------------
# Step 6: Predictions
# -------------------------
y_pred = knn.predict(X_test)

# -------------------------
# Step 7: Accuracy
# -------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n=======================")
print("KNN MODEL RESULTS")
print("=======================")
print("Accuracy:", accuracy * 100, "%")