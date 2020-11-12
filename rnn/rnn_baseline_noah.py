# Import Libraries
import numpy as np
from collections import defaultdict
from pydash import _
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN
import matplotlib.pyplot as plt
import os
from random import randint
from tensorflow.keras.callbacks import EarlyStopping


# Functions to load and process the data
def get_label(index):
  return index <= 20

def trim(arr,length):
  arr = arr[:length,:]
  return arr

def window(a, w = 4, o = 2, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

def get_dataset(overlap=5, window_size=10, time_steps=20, language=None, max_len=80):
  if language:
    dataset = _.filter(processed, lambda x: x['language'] == language)
  else:
    dataset = processed
  num_features = 7 # number of features on each time step
  # x,y,meta = [],[],[]
  genuine = forged = None
  if not max_len:
    pass
  else:
    for val in dataset:
      update = False
      g = val['values'][:20]
      f = val['values'][20:]

      g = np.array(_.map(_.filter(g, lambda i: len(i) >= max_len), lambda x: window(trim(x,max_len).reshape(-1),window_size*num_features,overlap*num_features,True)))
      f = np.array(_.map(_.filter(f, lambda i: len(i) >= max_len), lambda x: window(trim(x,max_len).reshape(-1),window_size*num_features,overlap*num_features,True)))
      if genuine is None and g.size > 0:
        genuine = g
        update = True
      if forged is None and f.size > 0:
        forged = f
        update = True
        continue
      if (g.size > 0 or f.size > 0) and not update:
        genuine = genuine if g.size == 0 else np.append(genuine,g,axis=0)
        forged = forged if f.size == 0 else np.append(forged,f,axis=0)

    print(genuine.shape) # (num_samples, num_time_steps, window_size*num_features)
    print(f'Options\nWindow size --> {window_size}\nOverlap (in samples) --> {overlap}\nMax length {max_len}\n{"— "*20}')
    print(f'Number of time steps --> {genuine.shape[1]}\nNumber of features at each step --> {genuine.shape[2]}')
    print(f'Number of genuine samples --> {genuine.shape[0]}')
    print(f'Number of forged samples --> {forged.shape[0]}')
    x = { "genuine": genuine,
          'forged': forged }
    return x


# Parameters for lodaing and processing the data
DATASET_PATH = '/content/drive/My Drive/Colab Notebooks/Signatures_ECE765/datasets/Task2/'
ENGLISH = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]

window_size = 3
overlap = 2
max_len = 80
timesteps = 20
language = None


# Load and process the data
langs = ['chinese', 'english']
processed = []
for i in range(1,41):
    values = np.array([])
    # _.for_each(list(range(1,41)), lambda x: values.append(np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1)))
    # values = _.reduce(list(range(1,41)), lambda total, x: np.append(values, np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1)))
    # for x in range(1,41):
    #   np.append(values, )
    # [np.append(values, np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT',skiprows=1) for x in range(1,41)]
    processed.append({ 'user': i,
                       'values': [np.loadtxt(f'{DATASET_PATH}/U{i}S{x}.TXT', skiprows=1) for x in range(1,41)],
                      #  'values': [np.loadtxt('{}/U{}S{}.TXT'.format(DATASET_PATH,i,x), skiprows=1) for x in range(1,41)],
                       'language': 'english' if i in ENGLISH else 'chinese' })

min_length = defaultdict(lambda:float('inf'))
max_length = defaultdict(lambda:float('-inf'))
for lang in langs:
  for val in _.filter(processed, lambda x: x['language'] == lang):
    curr_min = min(len(x) for x in val['values'])
    curr_max = max(len(x) for x in val['values'])
    if curr_min < min_length[lang]:
      min_length[lang] = curr_min
    if curr_max > min_length[lang]:
      max_length[lang] = curr_max

print(f'{len(processed)} signatures loaded\n\n')
print(f'English min length: {min_length["english"]}\nChinese min length: {min_length["chinese"]}\n\n')
print(f'English max length: {max_length["english"]}\nChinese max length: {max_length["chinese"]}')

data = get_dataset(overlap=overlap, window_size=window_size, time_steps=20, language=language, max_len=(100 if language=="english" else 80))

x_i = []
y_i = []
x_it = []
y_it = []
test_count = defaultdict(int)
tot = 80

''' Below is Avery's code, which alternates genuine and forged'''
for i in range(0, 2*len(x['genuine'])):
  if i % 2 == 0:
    if test_count['genuine'] < tot:
      x_it.append(x['genuine'][i//2])
      y_it.append([1])
      test_count['genuine'] += 1
    else:
      x_i.append(x['genuine'][i//2])
      y_i.append([1])
  else:
    if test_count['forged'] < tot:
      x_it.append(x['forged'][i//2])
      y_it.append([0])
      test_count['forged'] += 1
    else:
      x_i.append(x['forged'][i//2])
      y_i.append([0])

''' Below is Noah's original code, which puts all the genuine then all the forged'''
# for val in ['genuine','forged']:
#   for i in x[val]:
#     if val == 'genuine' and test_count['genuine'] < tot:
#       x_it.append(i)
#       y_it.append([1])
#       test_count['genuine'] += 1
#     elif val == 'forged' and test_count['forged'] < tot:
#       x_it.append(i)
#       y_it.append([0])
#       test_count['forged'] += 1
#     else:
#       x_i.append(i)
#       y_i.append([0] if val == 'genuine' else [1])

print(len(x_i))
print(x_i[0].shape)
print(len(y_i))
print(len(x_it))
print(x_it[0].shape)
print(len(y_it))

x_train = np.zeros((len(x_i), x_i[0].shape[0], x_i[0].shape[1]))
for i in range(len(x_i)):
  x_train[i] = x_i[i]

x_valid = np.zeros((len(x_it), x_it[0].shape[0], x_it[0].shape[1]))
for i in range(len(x_it)):
  x_valid[i] = x_it[i]

y_train = np.zeros((len(y_i)))
for i in range(len(y_i)):
  y_train[i] = y_i[i][0]

y_valid = np.zeros((len(y_it)))
for i in range(len(y_it)):
  y_valid[i] = y_it[i][0]

print(x_train)
print(x_valid)
print(y_valid)
print(len(y_valid))
print(y_train)
print(len(y_train))


# Simple RNN Model
model_RNN = Sequential()
model_RNN.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.5))
model_RNN.add(Dense(1, activation='sigmoid'))
model_RNN.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_RNN.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_RNN.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, callbacks=[early_stopping])


# LSTM Model
model_LSTM = Sequential()
model_LSTM.add(LSTM(10, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.5))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_LSTM.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_LSTM.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, callbacks=[early_stopping])
