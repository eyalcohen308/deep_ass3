import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

UNKNOWN_CHAR = '*'


def tensorize_sentence(sentence, F2I):
	"""
	Convert sentence to indexed tensor representation matrix.
	:param sentence: a sequence.
	:param F2I: feature to index dictionary
	:return: indexed tensor representation matrix
	"""
	tensor_vector = torch.zeros(len(sentence), len(F2I))
	for i, feature in enumerate(sentence):
		one_hot_vec = torch.zeros(len(F2I))
		one_hot_vec[F2I[feature]] = 1
		tensor_vector[i] = one_hot_vec
	return tensor_vector


def get_file_directory(data_kind):
	"""
	return the directory of the file from data directory.
	:param data_kind: name of the data file
	:return: local directory.
	"""
	return "./data/{0}".format(data_kind)


def parse_data_from_file_dir(data_dir, is_test=False):
	"""
	Get's data directory, parse it's content with '\t' to sequence and tag.
	if is_test is true, skip on creating tags.
	:param data_dir: data location.
	:param is_test: imply whether to tag or not
	:return: dataset (sequence, [int tag])
	"""
	data_set = []
	lines_list = open(data_dir).read().splitlines()
	for line in lines_list:
		split_line = line.split('\t')
		sentence = split_line[0]
		if not is_test:
			current_tag = split_line[1]
			data_set.append((sentence, [int(current_tag)]))
		else:
			data_set.append(sentence)
	return data_set
