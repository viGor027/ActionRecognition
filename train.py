from src.data import train_ds, val_ds
from src.model.model import model
import tensorflow as tf

# tensorboard_callback = tf.keras.callbacks.TensorBoard('tensorboard/')
es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
]

adam = tf.keras.optimizers.Adam(learning_rate=(0.001 / 10))

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=metrics)

model.fit(train_ds, validation_data=val_ds, callbacks=[es], epochs=200)
model.save('model.keras')
model.save('model.h5')