"""
Sequence Classification Model using MobileNet Feature Extractor

This script defines a TensorFlow model for sequence classification using the MobileNet feature extractor.

Module Dependencies:
    - tensorflow (imported as tf)
    - src.constants (imported as CLASSES, IMAGE_SIZE, N_FRAMES)

Global Constants:
    - CLASSES (list): List of class labels for the classification task.
    - IMAGE_SIZE (int): Size of the input images (height, width, channels).
    - N_FRAMES (int): Number of frames in each sequence.

Model Architecture:
    - MobileNet Feature Extractor:
        - Utilizes the MobileNet architecture pre-trained on ImageNet.
        - Input shape: (IMAGE_SIZE, IMAGE_SIZE, 3).
        - Output: Global average pooling layer.

    - Sequence Classification:
        - Takes input sequences with shape (N_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3).
        - Applies TimeDistributed layer to the MobileNet feature extractor.
        - Adds TimeDistributed Flatten layer to flatten the spatial dimensions.
        - Applies GRU (Gated Recurrent Unit) layers for sequence modeling.
        - Final Dense layer with softmax activation for classification.

    - Model Input and Output:
        - Input: Sequences of image frames with shape (N_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3).
        - Output: Predicted class probabilities for each sequence.

Example Usage:
    - Use this script as a template for creating a sequence classification model.
    - Adjust hyperparameters and input shapes based on the specific task.
"""

import tensorflow as tf
from src.constants import CLASSES, IMAGE_SIZE, N_FRAMES


input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
mobilenet = tf.keras.applications.MobileNet(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet',
    pooling='avg',
)


input_sequence = tf.keras.layers.Input((N_FRAMES,) + input_shape)
sequence_embedding = tf.keras.layers.TimeDistributed(mobilenet)
outputs = sequence_embedding(input_sequence)
new_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(outputs)
new_output = tf.keras.layers.GRU(units=64, return_sequences=True)(new_output)
new_output = tf.keras.layers.GRU(units=64)(new_output)
new_output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(new_output)
model = tf.keras.Model(inputs=input_sequence, outputs=new_output)
