from sklearn.preprocessing import LabelEncoder
#from keras.models import model_from_json
import tensorflowjs as tfjs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras.models import load_model
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='./dataset',
               help="path to input dataset")
ap.add_argument("-l", "--le", type=str, default='./le.pickle',
                help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="confusion_matrix.png",
 help="path to confusion matrix")
args = vars(ap.parse_args())

bs = 8

print("Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)



model = load_model('LivenessNet.model')

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                test_size=0.25, random_state=42)

#testY_predict = model.predict(testY, batch_size=bs)

#cm = confusion_matrix(np.asarray(testY).argmax(axis=1), np.asarray(testY_predict).argmax(axis=1))
labels = ['fake', 'real']

# Make predictions on test data
predictions = model.predict(testX, batch_size=bs)

# Convert one-hot encoded labels back to categorical labels
y_true = np.argmax(testY, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


plt.figure(figsize=(10,10))
sns.heatmap(cm_norm, annot=True, fmt=".2%", linewidths=0.5, square = True, cmap='Blues',xticklabels = labels, yticklabels=labels)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(args["plot"])