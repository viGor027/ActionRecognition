"""
Script for loading a pre-trained model and generating confusion matrices for test set predictions.

Requirements:
    - TensorFlow (tf)
    - Numpy (np)
    - scikit-learn.metrics.ConfusionMatrixDisplay
    - matplotlib.pyplot

Usage:
    - Ensure that the 'model.h5' file is present in the current working directory.
    - Ensure that the 'get_gen' function is correctly implemented in the 'src.data' module.
"""

import tensorflow as tf
from src.data import get_gen
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model.h5')

y_pred = []
y_true = []

for video, label in get_gen('test')():
    prediction = model.predict(video.reshape((1, *video.shape)))
    y_pred.append(np.argmax(prediction[0]))
    y_true.append(label[0])

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axs[0])
axs[0].set_title("Confusion matrix")
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=axs[1],
                                        normalize="true", values_format=".0%")
axs[1].set_title("Percentage Error")
plt.savefig("confusion_matrices")
plt.show()
