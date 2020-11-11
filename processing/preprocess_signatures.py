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


def normalize_data(data, skipcols=None):
  """Normalizes the data so that all features are in the range [0,1]"""
  rows = 0
  for i in range(len(data)):
    rows += data[i].shape[0]

  rows2 = 0
  data_all = np.empty((rows, data[0].shape[1]))
  for i in range(len(data)):
    data_all[rows2:rows2+data[i].shape[0], :] = data[i]
    rows2 += data[i].shape[0]

  data_norm = []
  for i in range(len(data)):
    data_norm_sig = np.zeros((data[i].shape[0], data[i].shape[1]))
    for f in range(data[i].shape[1]):
      if f in skipcols:
        data_norm_sig[:, f] = data[i][:, f]
      else:
        data_norm_sig[:, f] = (data[i][:, f] - data_all[:, f].min()) / data_all[:, f].ptp()
    data_norm.append(data_norm_sig)

  return data_norm


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


def split_train_test(x, y, names, train_percentage=0.75):
	"""Randomly splitting the data to train and test sets (by IDs) using the given percentage split."""
	np.random.seed(0)  # Setting this for reproducible results
	subjects = [i.split('S')[0] for i in names]
	unique_subjects = list(set(subjects))
	train_subjects = np.random.choice(unique_subjects, size=int(len(unique_subjects)*train_percentage), replace=False)
	train_indices = [i for i, e in enumerate(subjects) if e in train_subjects]
	test_indices = [i for i, e in enumerate(subjects) if e not in train_subjects]

	x_train = x[train_indices]
	y_train = y[train_indices]
	x_test = x[test_indices]
	y_test = y[test_indices]

	return x_train, x_test, y_train, y_test


if __name__ == "__main__":
	INPUT_PATH = '../datasets/Task2/'
	OUTPUT_PATH = '../datasets/processed'

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
