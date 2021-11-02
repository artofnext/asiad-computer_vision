import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt


###################################################################
#
# Neste exemplo o dataset e' carregado a partir do sistema de ficheiros
# e apenas e' dividido em treino e validacao

batch_size = 32
img_height = 180
img_width = 180
dataset_path = "../img/flower_photos"

train_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="training",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  labels='inferred',
  label_mode = 'categorical',
  validation_split=0.2,
  subset="validation",
  seed=1234,
  image_size=(img_height, img_width),
  batch_size=batch_size)

labels = train_ds.class_names
print(labels)

plt.figure(1, figsize=(10, 10))
for x_batch, y_batch in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i].numpy().astype("uint8"))
        plt.title(labels[np.argmax(y_batch[i,:])])
        plt.axis("off")
plt.show()

train_ds = train_ds.cache()
val_ds = val_ds.cache()

num_classes = 5

model = tf.keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.Conv2D(16, 5, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

model.summary()

# epochs = 5
epochs=3
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)



plt.figure(2, figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()