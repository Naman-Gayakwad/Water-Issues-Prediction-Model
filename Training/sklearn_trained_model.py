import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# prepare data

img2vec = Img2Vec()

data_dir = 'C:/D drive/code/code/Water Issues Prediction Model/Training/data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_).convert('RGB')

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

# train model

model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

# test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix
cm = confusion_matrix(data['validation_labels'], y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(data['validation_labels']))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plot accuracy trend (example values)
dataset_sizes = [200, 400, 600, 800, 1000]
accuracies = [75, 82, 85, 88, 90]
plt.plot(dataset_sizes, accuracies, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Trend with Increasing Dataset Size')
plt.grid(True)
plt.show()

print(score*100,'this is my accuracy')

# save the model

with open('./sklearn_model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()