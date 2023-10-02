import tensorflow as tf
from data import test_ds


model = tf.keras.models.load_model('model.h5')

model.evaluate(test_ds)
