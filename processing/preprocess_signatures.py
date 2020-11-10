import os
import numpy as np


def read_data(input_path, english_only, english_indices=None):
	"""Returns the list of signatures, labels, and ids. These can have English-only signatures if specified."""
	if english_indices is None:
		english_indices = []
	sig_list = []
	lab_list = []		# Genuine = 0, Forged = 1
	id_list = []
	if english_only:
		input_files = [i for i in sorted(os.listdir(input_path)) if int(i.split('S')[0].replace('U','')) in english_indices]
	else:
		input_files = sorted(os.listdir(input_path))

	for file in input_files:
		data = np.genfromtxt(input_path + file, skip_header=1)
		data = np.delete(data, 2, axis=1)		# drop the timestamp column
		try:
			sig_list.append(data)
			id_list.append(file.split('.')[0])
			if int(file.split('S')[1].replace('.TXT', '')) < 21:
				lab_list.append(0)		# Genuine
			else:
				lab_list.append(1)		# Forged
		except:
			print(file)

	return sig_list, lab_list, id_list


def merge_timesteps(x, timesteps_to_merge):
	"""Combines multiple timesteps of raw signature data into a single timestep.
	E.g., if timesteps_to_merge is 3, then each 3 rows will now be concatenated into 1,
	meaning there will be 3*num_features in the row."""
	x_merged = []
	for x_i in x:
		x_i_merged = []
		start_index = 0
		end_index = timesteps_to_merge
		while end_index < x_i.shape[0]:
			x_i_merged.append(np.concatenate(x_i[start_index:end_index]))
			start_index = start_index + timesteps_to_merge
			end_index = end_index + timesteps_to_merge

		# Accounting for the very last part of the sequence
		x_i_merged.append(np.concatenate(x_i[x_i.shape[0] - timesteps_to_merge:x_i.shape[0], :]))
		x_merged.append(np.array(x_i_merged))

	return x_merged


def split_sequences(x, y, names, window_length, window_stride):
	"""Splits all input sequences into subsequences by moving along a window of given window_length and stride."""
	split_x = []
	split_y = []
	split_ids = []
	for (x_i, y_i, id_i) in zip(x, y, names):
		start_index = 0
		end_index = window_length
		while end_index < x_i.shape[0]:
			split_x.append(x_i[start_index:end_index, :])
			split_y.append(y_i)
			split_ids.append(id_i)
			start_index = start_index + window_stride
			end_index = end_index + window_stride

		# Accounting for the very last part of the sequence
		split_x.append(x_i[x_i.shape[0]-window_length:x_i.shape[0], :])
		split_y.append(y_i)
		split_ids.append(id_i)

	return np.array(split_x), np.array(split_y), np.array(split_ids)


if __name__ == "__main__":
	INPUT_PATH = '../datasets/Task2/'
	OUTPUT_PATH = '../datasets/processed'

	eng_indices = [2, 4, 6, 8, 10, 12, 13, 15, 18, 20, 22, 24, 25, 28, 30, 32, 33, 34, 35, 40]  # English signature numbers
	eng_only = True		# Whether to only consider English or English and Chinese signatures
	stride = 20			# How far to move the window for creating fixed-length subsequences with each signature
	length = 25			# How big each window is for the fixed-length sequences
	merge_num = 3		# How many rows to concatenate into a single row -- see function for more details

	signatures, labels, ids = read_data(INPUT_PATH, english_only=eng_only, english_indices=eng_indices)
	signatures_merged = merge_timesteps(x=signatures, timesteps_to_merge=merge_num)
	x_subsequences, y_subsequences, ids_subsequences = split_sequences(x=signatures_merged, y=labels, names=ids, window_length=length, window_stride=stride)
