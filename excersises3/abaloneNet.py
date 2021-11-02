import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
# Leitura dos datasets
abalone_train = pd.read_csv('abalone_train.csv',
                             names=["Length", "Diameter", "Height", "Whole weight",
                                    "Shucked weight", "Viscera weight", "Shell weight", "Age"])
abalone_test = pd.read_csv('abalone_test.csv',
                            names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
                                   "Viscera weight", "Shell weight", "Age"])

# Preparação dos dados
abalone_train_features = abalone_train
abalone_train_labels = abalone_train_features.pop("Age")
abalone_test_features = abalone_test
abalone_test_labels = abalone_test_features.pop("Age")

x_train = np.array(abalone_train_features)
y_train = np.array(abalone_train_labels)
x_test = np.array(abalone_test_features)
y_test = np.array(abalone_test_labels)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


shellModel = tf.keras.Sequential([
    layers.Dense(14, activation='sigmoid', input_shape=(1,7)),
    # layers.Dense(20, activation='sigmoid'),
    layers.Dense(1)
])

# shellModel.summary()

shellModel.compile(loss = tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.SGD(learning_rate=0.1),
                  # optimizer=tf.optimizers.Adam(learning_rate=0.001),
                  )

history = shellModel.fit(x_train, y_train, batch_size=128, epochs=500, validation_data=(x_test, y_test))

y_pred = np.array(shellModel(x_test)).astype(int)

# print(output_pred)

plt.figure(1)
plt.plot(y_test, y_pred, 'bo', alpha=.1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model')
plt.ylabel('Predicted')
plt.xlabel('Real')
# plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

plt.show()