#### Libraries
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import regularizers

learning_rate = 1.5
epochs = 25
batch_size = 10

dataset=mnist.load_data()

dat=np.array(dataset)
(x_train, y_train), (x_test, y_test) = dataset

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255
x_testv /= 255

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(350, activation='sigmoid', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate, momentum=0.1),metrics=['accuracy'])

history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )