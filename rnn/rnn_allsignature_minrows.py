# Import Libraries
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN
import matplotlib.pyplot as plt
import os


# Load the data
DATA_PATH = '../datasets/Task2/'

min_rows = 9999
for file in sorted(os.listdir(DATA_PATH)):
  f = open(DATA_PATH + file, 'r')
  rows = int(f.readline())
  if rows < min_rows:
    min_rows = rows
  print('{0}: {1} -> MIN ROWS = {2}'.format(file, rows, min_rows))

print(min_rows)

genuine = []
forgery = []
for file in sorted(os.listdir(DATA_PATH)):
  data = np.genfromtxt(DATA_PATH + file, skip_header=1, max_rows=min_rows)
  data = np.delete(data, 2, axis=1)  # drop the timestamp column
  # print(data)
  # print(data.shape)
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
print(len(forgery))


# Split data into train and valid
x_train = np.zeros((1200, min_rows, 6))
x_valid = np.zeros((400, min_rows, 6))
y_train = np.zeros((1200, 1))
y_valid = np.zeros((400, 1))
for i in range(0, 2*len(genuine)):
  if i % 2 == 0:
    if i < 1200:
      x_train[i] = genuine[i//2]
      y_train[i] = 1
    else:
      x_valid[i-1200] = genuine[i//2]
      y_valid[i-1200] = 1
  else:
    if i < 1200:
      x_train[i] = forgery[i//2]
      y_train[i] = 0
    else:
      x_valid[i-1200] = forgery[i//2]
      y_valid[i-1200] = 0
print(x_train)
print(x_valid)
print(y_train)
print(y_valid)


# Train Simple RNN model
model_RNN = Sequential()
model_RNN.add(SimpleRNN(500))
model_RNN.add(Dense(1, activation='sigmoid'))
# model_RNN.add(Dropout(rate=0.2))
model_RNN.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# model_RNN.summary()

model_RNN.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20)


# Train LSTM model
model_LSTM = Sequential()
model_LSTM.add(LSTM(100))
model_LSTM.add(Dense(1, activation='sigmoid'))
# model_RNN.add(Dropout(rate=0.2))
model_LSTM.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
# print(model_LSTM.summary)

model_LSTM.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=20)
