"""Script used for evaluation of a model"""

import tensorflow as tf
from src.data import test_ds


model = tf.keras.models.load_model('../model.h5')

model.evaluate(test_ds)
