# Water Issues Prediction Model

This repository contains a **machine learning model** designed to predict water-related issues such as **urban floods, drainage blockages, and sanitation problems** using image analysis.

## 📁 Repository Structure

```
Water-Issues-Prediction-Model/
│-- convert_model.py                # Converts Keras model to TFLite format
│-- infer.py                        # Runs inference using the trained model
│-- predict_tflite.py               # Runs inference using the TFLite model
│
├── Model/
│   │-- model_tf.keras              # Trained Keras model
│   │-- model.tflite                # Converted TFLite model
│   │-- sklearn_model.p             # Scikit-learn trained model
│
├── Prediction Using Model/
│   │-- Sample Images/
│   │   │-- dirtylaketest.jpg
│   │   │-- drainagehole.jpg
│   │   │-- drainagetest.jpeg
│   │   │-- flood_test.jpg
│
├── Training/
│   │-- data/
│   │   │-- train/                  # Training dataset
│   │   │-- val/                    # Validation dataset
│   │-- sklearn_trained_model.py    # Sklearn training script
│   │-- tflitemodel.py              # Converts the trained model to TFLite
```

## 🚀 Features
- Predicts water-related issues from images.
- Uses **Scikit-Learn, Keras**, and **TFLite** for efficient model deployment.
- Includes both **standard and lightweight models (TFLite)** for mobile-friendly inference.
- Provides scripts for training, conversion, and prediction.

## 📌 Installation
To set up the environment, install the required dependencies:
```sh
pip install -r requirements.txt
```

## 📊 Model Training
To train the model using **Scikit-Learn**:
```sh
python Training/sklearn_trained_model.py
```

To train and convert the model to **TFLite**:
```sh
python Training/tflitemodel.py
```

## 🔄 Model Conversion
To convert the Keras model to **TFLite** format:
```sh
python convert_model.py
```

## 🏗️ Running Inference
Run inference using the standard model:
```sh
python infer.py --image_path Sample Images/dirtylaketest.jpg
```
Run inference using the **TFLite model**:
```sh
python predict_tflite.py --image_path Sample Images/dirtylaketest.jpg
```

## 📢 Contributing
Feel free to submit issues and pull requests to improve this project!

## 🔗 Repository Link
[Water Issues Prediction Model](https://github.com/Naman-Gayakwad/Water-Issues-Prediction-Model)

---
📌 **Author:** Naman Gayakwad  
📩 **Contact:** namangayakwad089@gmail.com  
📜 **License:** MIT
