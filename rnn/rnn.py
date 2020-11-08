# Import Libraries
import numpy as np
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN
import matplotlib.pyplot as plt
import os
from random import randint
from tensorflow.keras.callbacks import EarlyStopping


# Adjustable parameters
DATA_PATH = '../datasets/Task2/'
include = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]  # English signature numbers
features = 6
random_start = False  # True = use random signature window of size min_rows, False = start signature window of size min_rows at time zero
english_only = True  # Whether to only consider English or English and Chinese signatures
normalize = True  # Whether to notmalize the data befroe it is fed to the network
window = True  # Whether to use windows of data as input to the models


# Load the data
min_rows = 9999
for file in sorted(os.listdir(DATA_PATH)):
  if not english_only or int(file.split('S')[0].replace('U', '')) in include:
    f = open(DATA_PATH + file, 'r')
    rows = int(f.readline())
    if rows < min_rows:
      min_rows = rows
    print('{0}: {1} -> MIN ROWS = {2}'.format(file, rows, min_rows))
# print(min_rows)

genuine = []
forgery = []
for file in sorted(os.listdir(DATA_PATH)):
  if not english_only or int(file.split('S')[0].replace('U', '')) in include:
    if random_start:
      data = np.genfromtxt(DATA_PATH + file, skip_header=1)
      data = np.delete(data, 2, axis=1)  # drop the timestamp column
      start_idx = randint(0, len(data) - min_rows)
      print(start_idx)
      # print(data)
      # print(data.shape)
      try:
        if int(file.split('S')[1].replace('.TXT', '')) < 21:
          # print('genuine')
          genuine.append(data[start_idx:start_idx+min_rows])
        else:
          # print('forgery')
          forgery.append(data[start_idx:start_idx+min_rows])
      except:
        print(file)
    else:
      data = np.genfromtxt(DATA_PATH + file, skip_header=1, max_rows=min_rows)
      data = np.delete(data, 2, axis=1)  # drop the timestamp column
      try:
        if int(file.split('S')[1].replace('.TXT', '')) < 21:
          # print('genuine')
          genuine.append(data)
        else:
          # print('forgery')
          forgery.append(data)
      except:
        print(file)
print(len(genuine))
# print(genuine)
print(len(forgery))
# print(forgery)


# Window the data
if window:
  samples = 10
  stride = 7
  kdelta = samples//2
  genuine2 = []
  forgery2 = []

  for sig in genuine:
    data = np.zeros((min_rows//stride, features*samples))
    i = 5
    for row in data:
      for j in range(kdelta*-1, kdelta):
        k = (j+kdelta)*features
        row[k:k+features] = sig[i+j]
      i += stride
    genuine2.append(data)

  for sig in forgery:
    data = np.zeros((min_rows//stride, features*samples))
    i = 5
    for row in data:
      for j in range(kdelta*-1, kdelta):
        k = (j+kdelta)*features
        row[k:k+features] = sig[i+j]
      i += stride
    forgery2.append(data)

genuine = genuine2
print(genuine[0].shape)
forgery = forgery2
print(forgery[0].shape)


# Split data into train and valid
len_train = 600 if english_only else 1200
len_valid = 200 if english_only else 400

x_train = np.zeros((len_train, min_rows, features)) if not window else np.zeros((len_train, min_rows//stride, features*samples))
x_valid = np.zeros((len_valid, min_rows, features)) if not window else np.zeros((len_valid, min_rows//stride, features*samples))
y_train = np.zeros((len_train, 1))
y_valid = np.zeros((len_valid, 1))
for i in range(0, 2*len(genuine)):
  if i % 2 == 0:
    if i < len_train:
      x_train[i] = genuine[i//2]
      y_train[i] = 1
    else:
      x_valid[i-len_train] = genuine[i//2]
      y_valid[i-len_train] = 1
  else:
    if i < len_train:
      x_train[i] = forgery[i//2]
      y_train[i] = 0
    else:
      x_valid[i-len_train] = forgery[i//2]
      y_valid[i-len_train] = 0
# print(x_train)
# print(x_valid)
# print(y_train)
# print(y_valid)


# Normalize the data
if normalize:
  x_all = np.zeros((len_train+len_valid, min_rows, features)) if not window else np.zeros((len_train+len_valid, min_rows//stride, features*samples))
  for i in range(0, 2*len(genuine)):
    if i % 2 == 0:
      x_all[i] = genuine[i//2]
    else:
      x_all[i] = forgery[i//2]
  print(x_all.shape)

  for i in range(samples*features):
    if i != 2:
      x_train[:, :, i] =  (x_train[:, :, i] - x_all[:, :, i].min()) / x_all[:, :, i].ptp()
      x_valid[:, :, i] =  (x_valid[:, :, i] - x_all[:, :, i].min()) / x_all[:, :, i].ptp()

  print(x_train.shape)
  print(x_valid.shape)
  # print(x_train)


# Train Simple RNN model
model_RNN = Sequential()
if window:
  model_RNN.add(SimpleRNN(10, input_shape=(min_rows//stride, features*samples), dropout=0.5))
else:
  model_RNN.add(SimpleRNN(60, input_shape=(min_rows, features), dropout=0.5))
model_RNN.add(Dense(1, activation='sigmoid'))
model_RNN.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_RNN.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_RNN.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, callbacks=[early_stopping])


# Train LSTM model
model_LSTM = Sequential()
if window:
  model_LSTM.add(LSTM(10, input_shape=(min_rows//stride, features*samples), dropout=0.5))
else:
  model_LSTM.add(LSTM(60, input_shape=(min_rows, features), dropout=0.5))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model_LSTM.summary())
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

model_LSTM.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, callbacks=[early_stopping])
