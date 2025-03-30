import pickle

# Importing Img2Vec for converting images to feature vectors
from img2vec_pytorch import Img2Vec
# Importing PIL for image processing
from PIL import Image

# Loading the pre-trained model from a pickle file
with open('./model.p', 'rb') as f:
    model = pickle.load(f)

# Initializing the Img2Vec object for feature extraction
img2vec = Img2Vec()

# Path to the input image
image_path = 'Image path goes here'

# Opening the image using PIL
img = Image.open(image_path)

# Extracting feature vector from the image
features = img2vec.get_vec(img)

# Using the loaded model to make a prediction based on the extracted features
pred = model.predict([features])

# Printing the prediction result
print(pred)