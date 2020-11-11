import sys
sys.path.append('../')
from processing.preprocess_signatures import *

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping


# Load and Process the data
INPUT_PATH = '../datasets/Task2/'

eng_indices = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]  # English signature numbers
eng_only = True		# Whether to only consider English or English and Chinese signatures
stride = 20			# How far to move the window for creating fixed-length subsequences with each signature
length = 25			# How big each window is for the fixed-length sequences
merge_num = 3		# How many rows to concatenate into a single row -- see function for more details
train_test_split = 0.75		# This is how much of the data will be used for TRAINING, the rest is for testing (split by ID)
normalize = True 	# Whether you want to normalize the data or not

signatures, labels, ids = read_data(INPUT_PATH, english_only=eng_only, english_indices=eng_indices)
if normalize:
	signatures_normalized = normalize_data(signatures, skipcols=[2])
	signatures_merged = merge_timesteps(x=signatures_normalized, timesteps_to_merge=merge_num)
else:
	signatures_merged = merge_timesteps(x=signatures, timesteps_to_merge=merge_num)
signatures_subsequences, labels_subsequences, ids_subsequences = split_sequences(x=signatures_merged, y=labels, names=ids, window_length=length, window_stride=stride)
signatures_train, signatures_test, labels_train, labels_test = split_train_test(x=signatures_subsequences, y=labels_subsequences, names=ids_subsequences, train_percentage=0.75)


# SimpleRNN Model
model_RNN = Sequential()
model_RNN.add(SimpleRNN(10, input_shape=(length, 6*merge_num), dropout=0.1))
model_RNN.add(Dense(1, activation='sigmoid'))
model_RNN.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_RNN.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_RNN.fit(signatures_train, labels_train, validation_data=(signatures_test, labels_test), epochs=50, callbacks=[early_stopping])


# LSTM Model
model_LSTM = Sequential()
model_LSTM.add(LSTM(10, input_shape=(length, 6*merge_num), dropout=0.1))
model_LSTM.add(Dense(1, activation='sigmoid'))
model_LSTM.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
print(model_LSTM.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model_LSTM.fit(signatures_train, labels_train, validation_data=(signatures_test, labels_test), epochs=50, callbacks=[early_stopping])
