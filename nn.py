import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.python.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv('data/train.csv')
im = Image.open("data/train/000c8a36845c0208e833c79c1bffedd1.jpg")

shape = (len(df),) + im.size + (3,)

train_images = np.zeros(shape=shape, dtype=np.float16)
for i, file in enumerate(df['id']):
    train_images[i] = Image.open('data/train/' + str(file))

kernel_size=(3,3)
pool_size=(2,2)
first_filter=32
second_filter=64
third_filter=128

dropout_conv=0.3
dropout_dense=0.3

model = Sequential()
model.add(Conv2D(first_filter, kernel_size, padding='same', activation='relu', input_shape= (32,32,3)))
model.add(Conv2D(first_filter, kernel_size, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filter, kernel_size, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(second_filter, kernel_size, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filter, kernel_size, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filter, kernel_size, padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation = "sigmoid"))

model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])


earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)

model.fit(x=train_images,
          y=df['has_cactus'],
          batch_size=128,
          epochs=50,
          validation_split=0.2,
          callbacks=[reducel, earlystopper])

test_df = pd.read_csv('data/sample_submission.csv')

test_images = np.zeros(shape=(4000, 32, 32, 3), dtype=np.float16)
for i, file in enumerate(test_df['id']):
    test_images[i] = Image.open('data/test/' + str(file))

y_pred = model.predict(x=test_images)
test_df['has_cactus'] = y_pred
test_df.to_csv('submission.csv', index=False)
