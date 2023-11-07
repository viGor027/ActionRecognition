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
