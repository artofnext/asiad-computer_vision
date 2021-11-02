# Para desativar as mensgens de INFO e WARNINGS do tensorflow
import logging, os
logging.disable(logging.WARNING)
logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

##############################################################################################
# Preparacao do dataset
#
# Ajustar consoante se queira carregamento local do dataset (True)
# ou carregamento a partir das funçõe buitin do tensorflow/keras
# Caso se escolha carregamento local, ajustar os diretorios das imagens de treino e teste
LOCAL_DATASET = False
train_path = "mnist/training"
test_path = "mnist/test"

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
img_height = 28
img_width = 28
nClasses = len(labels)

if not LOCAL_DATASET:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train,(60000, img_height, img_width, 1))
    x_test = np.reshape(x_test,(10000, img_height, img_width, 1))
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)

else:
    batch_size = 60000
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        color_mode='grayscale',
        labels='inferred',
        label_mode = 'categorical',
        seed=12345,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    batch_size = 10000
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        color_mode='grayscale',
        labels='inferred',
        label_mode = 'categorical',
        seed=12345,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    for image, labels in train_ds:
        x_train = image.numpy()
        y_train = labels.numpy()

    for image, labels in test_ds:
        x_test = image.numpy()
        y_test = labels.numpy()

#######################################################################

# normalizacao
x_train = x_train / 255.0
x_test = x_test / 255.0

# split treino / validacao
split = x_train.shape[0] * 4 // 5  # 80%
x_val = x_train[split:]
y_val = y_train[split:]
x_train = x_train[:split]
y_train = y_train[:split]

# mostrar dimensoes das matrizes
print("Amostras de treino: " + str(x_train.shape))
print("Output de treino: " + str(y_train.shape))
print("Amostras de validacao: " + str(x_val.shape))
print("Output de validacao: " + str(y_val.shape))
print("Amostras de teste: " + str(x_test.shape))
print("Output de teste: " + str(y_test.shape))


###############################################################################
# Definicao, treino e teste do modelo
#
# definicao do modelo
model = tf.keras.models.Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.Dropout(0.2),
    layers.MaxPooling2D(pool_size=(2, 2), padding='valid'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.2),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

# sumario
model.summary()

# compilacao do modelo - escolha do algoritmo de otimizacao e funcao de perda
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

# treino do modelo
nEpochs = 15
history = model.fit(x_train, y_train, batch_size=32, epochs=nEpochs, validation_data=(x_val, y_val))

# obter predicoes e ground truth
output_pred = model(x_test)
y_pred = np.argmax(output_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

# calcular acertos no conjunto de teste
misses = np.count_nonzero(y_true-y_pred)
nTestSamples = y_true.shape[0]
accuracy = (nTestSamples - misses) / nTestSamples

print("Falhou {:d} de {:d} exemplos".format(misses, nTestSamples))
print("Taxa de acertos: {:.2f} %".format(accuracy * 100))

# gerar uma matriz de confusao
cm = confusion_matrix(y_true, y_pred)


########################################################################
# Mostrar figuras

# exemplos de imagens onde falhou
missesIdx = np.flatnonzero(y_true-y_pred)
plt.figure(1, figsize=(8, 8))
for i in range(0,16):
    idx = missesIdx[i]
    image = x_test[idx,:,:,:] * 255
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(image,cmap="gray")
    plt.title(labels[y_pred[idx]] + " (" + labels[y_true[idx]] + ")")
    plt.axis("off")

# evolucao da accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

# evolucao da loss
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

# matriz de confusao
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()
