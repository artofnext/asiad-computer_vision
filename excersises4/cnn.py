import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# comentar estas 6 linhas se não funcionar devido a problemas de rede
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train,(60000, 28, 28, 1))
x_test = np.reshape(x_test,(10000, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)
################################################################

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
img_height = 28
img_width = 28

################################################################
# descomentar as linhas que se seguem caso pretenda construir o dataset com base no sistema de ficheiros
# precisa de especificar o caminho para as pastas com os conjuntos de treino e teste

# train_path = "mnist/training"
# test_path = "mnist/training"
#
# batch_size = 60000
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   train_path,
#   color_mode='grayscale',
#   labels='inferred',
#   label_mode = 'categorical',
#   seed=1234,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# batch_size = 10000
# test_ds = tf.keras.utils.image_dataset_from_directory(
#   test_path,
#   color_mode='grayscale',
#   labels='inferred',
#   label_mode = 'categorical',
#   seed=1234,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# for image, labels in train_ds:
#     x_train = image.numpy()
#     y_train = labels.numpy()
#
# for image, labels in test_ds:
#     x_test = image.numpy()
#     y_test = labels.numpy()

#########################################################

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(x_train.shape[0])
print(x_train.shape[1])
print(x_train.shape[2])
print(x_train.shape[3])

# Desenvolver a partir daqui

model = tf.keras.models.Sequential([
layers.Conv2D(16, 5, padding=
'same', activation=
'relu', input_shape=(28, 28, 1)),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding=
'same', activation=
'relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding=
'same', activation=
'relu'),
layers.MaxPooling2D(),
layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128, activation=
'relu'),
layers.Dense(y_train.shape[1], activation=
"softmax")])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=2)