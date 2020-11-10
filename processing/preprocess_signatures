import os
import numpy as np


def split_sequences(x, y, names, length, stride):
	"""Splits all input sequences into subsequences by moving along a window of given length and stride."""
	split_x = []
	split_y = []
	split_ids = []
	for (x_i, y_i, id_i) in zip(x, y, names):
		start_index = 0
		end_index = length
		while end_index < x_i.shape[0]:
			split_x.append(x_i[start_index:end_index, :])
			split_y.append(y_i)
			split_ids.append(id_i)
			start_index = start_index + stride
			end_index = end_index + stride

		# Accounting for the very last part of the sequence
		split_x.append(x_i[x_i.shape[0]-length:x_i.shape[0], :])
		split_y.append(y_i)
		split_ids.append(id_i)

	return np.array(split_x), np.array(split_y), np.array(split_ids)


if __name__ == "__main__":
	INPUT_PATH = '../datasets/Task2/'
	OUTPUT_PATH = '../datasets/processed'

	english_indices = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]  # English signature numbers
	english_only = True		# Whether to only consider English or English and Chinese signatures
	stride = 40
	length = 50

	signatures = []
	labels = []		# Genuine = 0, Forged = 1
	ids = []
	if english_only:
		input_files = [i for i in sorted(os.listdir(INPUT_PATH)) if int(i.split('S')[0].replace('U','')) in english_indices]
	else:
		input_files = sorted(os.listdir(INPUT_PATH))

	for file in input_files:
		data = np.genfromtxt(INPUT_PATH + file, skip_header=1)
		data = np.delete(data, 2, axis=1)		# drop the timestamp column
		try:
			signatures.append(data)
			ids.append(file.split('.')[0])
			if int(file.split('S')[1].replace('.TXT', '')) < 21:
				labels.append(0)		# Genuine
			else:
				labels.append(1)		# Forged
		except:
			print(file)

	x_subsequences, y_subsequences, ids_subsequences = split_sequences(x=signatures, y=labels,
																	   names=ids, length=length, stride=stride)
