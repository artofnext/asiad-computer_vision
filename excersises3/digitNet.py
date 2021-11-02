import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Utilizar os datasets builtin do tensorflow - facilita a preparação dos dados
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalização dos valores de pixel para o intervalo [0 ... 1] - com imagens,
# este passo normalmente conduz a resultados bastante melhores
x_train = x_train / 255.0
x_test = x_test / 255.0

# Preparar a ground truth para o formato adequado, usando 10 classes
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

# Mostrar aas dimensões das matrizes para treino e teste
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Visualizar as primeiras 25 imagens do dataset
plt.figure(1)
fig, ax = plt.subplots(5,5)
for i in range(5):
    for j in range(5):
        ax[i,j].imshow(x_train[i*5+j], cmap=plt.get_cmap('gray'))

# A partir daqui é com vocês
