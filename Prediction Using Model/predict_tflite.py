import tensorflow as tf
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load Img2Vec for feature extraction
img2vec = Img2Vec()

# Load and preprocess the image
image_path = 'Image goes here'
img = Image.open(image_path).convert('RGB')

# Convert image to vector
features = img2vec.get_vec(img)

# Reshape to match model input shape
features = np.array([features], dtype=np.float32)  # Ensure dtype is float32

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], features)

# Run inference
interpreter.invoke()

# Get output predictions
predictions = interpreter.get_tensor(output_details[0]['index'])

# Get the class with the highest probability
predicted_class = np.argmax(predictions)

print("Predicted Class Index:", predicted_class)
