from flask import Flask, request, jsonify
import numpy as np
import tensorflow.lite as tflite
from PIL import Image
import io
import logging
from img2vec_pytorch import Img2Vec

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load TFLite model with error handling
try:
    interpreter = tflite.Interpreter(model_path="C:/D drive/code/code/Water Issues Prediction Model/Model/model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    interpreter = None

# Load Img2Vec for feature extraction
img2vec = Img2Vec()

# Function to preprocess the image using Img2Vec
def preprocess_image(image):
    try:
        image = image.convert('RGB')  # Ensure 3 channels
        features = img2vec.get_vec(image)  # Extract features using Img2Vec
        features = np.array([features], dtype=np.float32)  # Ensure dtype is float32
        logging.info(f"Processed feature vector shape: {features.shape}")  # Log feature vector shape
        return features
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model failed to load"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        # Load the image from the uploaded file
        image = Image.open(io.BytesIO(file.read()))
        input_data = preprocess_image(image)
        if input_data is None:
            return jsonify({"error": "Image processing failed"}), 400
        
        # Log input details for debugging
        logging.info(f"Input details: {input_details}")
        logging.info(f"Input data shape: {input_data.shape}")
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Log output data for debugging
        logging.info(f"Output data: {output_data}")
        
        # Example labels (update according to your model)
        labels = ["drainage_Blockage", "drainage_hole", "lake_and_pond", "urban_flooding"]
        
        # Get the predicted class index and corresponding label
        predicted_class_index = np.argmax(output_data)
        prediction = labels[predicted_class_index]

        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Failed to process image"}), 500

@app.route('/routes', methods=['GET'])
def get_routes():
    return jsonify([rule.rule for rule in app.url_map.iter_rules()])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
