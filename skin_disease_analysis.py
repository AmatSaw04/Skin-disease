# -*- coding: utf-8 -*-


# Install Kaggle
!pip install kaggle --quiet

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy kaggle.json from Drive (Change the path if needed)
!mkdir -p ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # Set correct permissions

# Download HAM10000 dataset
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip the dataset
!unzip -q skin-cancer-mnist-ham10000.zip -d skin_cancer_data

# Verify dataset extraction
import os
print("Dataset files:", os.listdir("skin_cancer_data"))

# Load metadata
import pandas as pd
df = pd.read_csv("skin_cancer_data/HAM10000_metadata.csv")
print(df.head())  # Display first 5 rows of metadata

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

data_dir = "/content/skin_cancer_data/ham10000_images_part_1"
metadata_path = "/content/skin_cancer_data/HAM10000_metadata.csv"

df = pd.read_csv(metadata_path)
print(df.head())

IMG_SIZE = (224, 224)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Sorry Bro !!!!  Warning: Image not found or cannot be read -> {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img

images = []
labels = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(data_dir, row['image_id'] + ".jpg")
    img = load_and_preprocess_image(img_path)

    if img is None:
        continue

    images.append(img)
    labels.append(row['dx'])

images = np.array(images)
labels = np.array(labels)

encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

features = resnet_model.predict(images, batch_size=32, verbose=1)
print(f"Extracted Features Shape: {features.shape}")

csv_path = "/content/drive/MyDrive/HAM10000_features.csv"
np_path = "/content/drive/MyDrive/HAM10000_features.npy"
np.savetxt(csv_path, features, delimiter=",")
np.save(np_path, features)
print(f"Features saved as CSV: {csv_path} and NumPy: {np_path}")

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

svm_model_path = "/content/drive/MyDrive/HAM10000_SVM_model.pkl"
joblib.dump(svm_model, svm_model_path)
print(f"SVM model saved to: {svm_model_path}")

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

print(f"Total images successfully loaded: {len(images)}")
print(f" Total missing/corrupted images: {len(df) - len(images)}")

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_model.fit(X_train, y_train)

svm_model_path = "/content/drive/MyDrive/HAM10000_SVM_model.pkl"
joblib.dump(svm_model, svm_model_path)
print(f"SVM Model Saved: {svm_model_path}")

y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f" SVM Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

from tensorflow.keras import models, layers

for layer in resnet_model.layers[:-1]:
    layer.trainable = False

model = models.Sequential()
model.add(resnet_model)
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(len(np.unique(labels_encoded)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(images, labels_encoded, epochs=5, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(images, labels_encoded)
print(f"Fine-Tuned ResNet Accuracy: {accuracy * 100:.2f}%")

