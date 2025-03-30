import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.model_selection import train_test_split
import sys

# Change default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Prepare data
img2vec = Img2Vec()
data_dir = 'C:/D drive/code/code/Water Issues Prediction Model/Training/data'
train_dir = os.path.join(data_dir, 'train')

features, labels = [], []
class_mapping = {}  # Mapping class names to numerical labels

for i, category in enumerate(os.listdir(train_dir)):
    class_mapping[category] = i  # Assign numerical label
    for img_path in os.listdir(os.path.join(train_dir, category)):
        img_path_ = os.path.join(train_dir, category, img_path)
        img = Image.open(img_path_).convert('RGB')
        img_features = img2vec.get_vec(img)

        features.append(img_features)
        labels.append(i)

features = np.array(features)
labels = np.array(labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define a simple neural network
model = keras.Sequential([
    layers.Input(shape=(features.shape[1],)),  # Input layer
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_mapping), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on validation data
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the model in TensorFlow format
model.save("model_tf.keras")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Plot training & validation accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.show()

plot_training_history(history)

# Predictions
y_pred = np.argmax(model.predict(X_val), axis=1)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_mapping):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_mapping.keys(), yticklabels=class_mapping.keys())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_val, y_pred, class_mapping)

# Precision-Recall Curve
def plot_precision_recall(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

# Get probability scores for each class
y_probs = model.predict(X_val)
y_probs = y_probs[:, 1]  # Assuming binary classification, otherwise modify this for multi-class

from sklearn.preprocessing import label_binarize

# Binarize the labels for multi-class PR curve
y_val_bin = label_binarize(y_val, classes=np.arange(len(class_mapping)))

# Plot PR curve for each class
for i in range(len(class_mapping)):
    precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_probs[:, i])
    plt.plot(recall, precision, marker='.', label=f'Class {i}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Multi-Class)')
plt.legend()
plt.show()

