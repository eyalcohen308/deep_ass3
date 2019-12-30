import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import re

UNKNOWN_CHAR = '*'
# TODO: check what is unkown word.
UNKNOWN_WORD = 'UUUKKKK'


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


def get_part1_file_directory(data_kind):
	"""
	return the directory of the file from data directory.
	:param data_kind: name of the data file
	:return: local directory.
	"""
	return "./data/{0}".format(data_kind)


def get_part3_file_directory(data_name="pos", data_kind="train"):
	"""
	return the directory of the file from data directory.
	:param data_kind: name of the data file
	:param data_name weather is pos or ner.
	:return: local directory.
	"""
	return "./data/{0}/{1}".format(data_name, data_kind)


def part1_parser(data_dir, is_test=False):
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


def pos_ner_parser(data_dir, data_name="pos", data_kind="train", to_lower=False, convert_digits=False):
	"""
	Get's data directory, parse it's content with '\t' to sequence and tag.
	if is_test is true, skip on creating tags.
	:param data_dir: data location.
	:param data_kind: imply whether to tag or not.
	:param data_name imply whether is pos or ner.
	:param to_lower if lower case the letters
	:param convert_digits if convert digits to '*DG*'
	:return: dataset (sequence, [int tag])
	"""
	sentences_data_set = []
	current_sentence_list = []
	# parse by spaces if post, if ner parse by tab.
	delimiter = ' ' if data_name == "pos" else '\t'
	data_set = []
	lines_list = open(data_dir).read().splitlines()
	for line in lines_list:
		raw_splitted = line.split(delimiter)
		word = raw_splitted[0]
		if word != '':
			# convert all chars to lower case.
			if to_lower:
				word = word.lower()
			# if we want to convert each digit to be DG for similarity, '300' = '400'.
			if convert_digits:
				word = re.sub('[0-9]', '*DG*', word)

			if data_kind != "test":
				tag = raw_splitted[1]
				current_sentence_list.append((word, tag))
			else:
				current_sentence_list.append(word)
		else:
			# finished iterate over one single sentence:
			sentences_data_set.append(current_sentence_list.copy())
			# reset list for next sentence.
			current_sentence_list.clear()
	return sentences_data_set


def make_loader(data, batch_size):
	# split the tupled given data to x and y.
	x, y = zip(*data)
	x, y = torch.tensor(x), torch.tensor(y)
	return DataLoader(TensorDataset(x, y), batch_size, shuffle=True)
