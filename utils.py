import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import re
import pickle

UNKNOWN_CHAR = '*'
PAD = "*PAD*"
CHAR_PAD = "*CHAR_PAD*"
UNIQUE_WORD = "UUUKKKK"
ARTIFACTS_PATH = "./artifacts"
MODEL_FILE_NAME = "saved_model.pt"
DATASET_NAME = "dataset.pickle"
DATASET_DIR = os.path.join(ARTIFACTS_PATH, DATASET_NAME)
MODEL_DIR = os.path.join(ARTIFACTS_PATH, MODEL_FILE_NAME)


def tensorize_sequence(sequence, F2I):
	"""`
	Convert sentence to indexed tensor representation matrix.
	:param sequence: a sequence.
	:param F2I: feature to index dictionary
	:return: indexed tensor representation matrix
	"""
	tensor_vector = torch.zeros(len(sequence), len(F2I))
	for i, feature in enumerate(sequence):
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


def make_loader(data, F2I, L2I, batch_size):
	# split the tupled given data to x and y.
	max_sequence_len = max(len(tup[0]) for tup in data)
	x = torch.LongTensor([convert_to_padded_indexes(sentence, F2I, max_sequence_len) for sentence, _ in data])
	y = torch.LongTensor([convert_to_padded_indexes(tags, L2I, max_sequence_len) for _, tags in data])

	dataset = TensorDataset(x, y)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_size_without_pad(value, tensor_arr):
	# get the size witout pad. if it zero convert it to 1 ( The packing doe's not execpt zero as word size)
	return (tensor_arr.tolist().index(value) or 1) if value in tensor_arr else len(tensor_arr)


# TODO: find the last index that not pad.


def convert_to_padded_indexes(sequence, index_dict, max_len):
	indexed_list = list()
	for sub_sequence in sequence:
		indexed_list.append(index_dict[sub_sequence] if sub_sequence in index_dict else index_dict[UNIQUE_WORD])
	pad_index = index_dict[PAD]
	indexed_list.extend([pad_index] * (max_len - len(indexed_list)))

	return indexed_list


def get_max_word_size(keys_list):
	return len(max(keys_list, key=len))


def create_char_input(input, dicts):
	F2I, I2F, C2I = dicts.F2I, dicts.I2F, dicts.C2I
	max_word_len = get_max_word_size(list(F2I.keys()))

	# input shape is (batch_size, num_sequences)
	word_input = input.view(-1)
	# input shape is (batch_size * num_sequences)
	char_input = torch.zeros(len(word_input), max_word_len, dtype=torch.long)
	# words_length = []
	for i, idx in enumerate(word_input):
		word = I2F[int(idx)]
		if word != PAD:
			# words_length.append(len(word))
			char_input[i] = torch.LongTensor(convert_to_padded_indexes(word, C2I, max_word_len))
	# else:
	# 	# doesn't matter because in the word embedding rapper it will skip them.
	# 	words_length.append(1)
	return char_input


def make_prefix(index, dicts):
	I2F, P2I = dicts.I2F, dicts.P2I
	word = I2F[int(index)]
	prefix = word[:3]
	return P2I[prefix] if prefix in P2I else P2I[UNIQUE_WORD[:3]]


def make_suffix(index, dicts):
	I2F, S2I = dicts.I2F, dicts.S2I
	word = I2F[int(index)]
	prefix = word[-3:]
	return S2I[prefix] if prefix in S2I else S2I[UNIQUE_WORD[-3:]]


def make_prefix_suffix_input(input, dicts):
	# [[item + 1 for item in list] for list in list_of_lists]
	# input shape is (batch_size, num_sequences)
	# prefix_input = torch.LongTensor(input.shape)
	# suffix_input = torch.LongTensor(len(input), len(input[0]))
	prefixes = [[make_prefix(word, dicts) for word in sentence] for sentence in input]
	suffixes = [[make_suffix(word, dicts) for word in sentence] for sentence in input]
	prefix_input = torch.LongTensor(prefixes)
	suffix_input = torch.LongTensor(suffixes)
	return prefix_input, suffix_input


def save_model_and_dicts(model, data_set):
	model.save(MODEL_DIR)
	with open(DATASET_DIR, 'wb') as file:
		pickle.dump(data_set, file, pickle.HIGHEST_PROTOCOL)
	print('data set and model were saved successfully!')


def load_model_and_dicts(model):
	model.load(MODEL_DIR)
	with open(DATASET_DIR, 'rb') as file:
		data_set = pickle.load(file)
	print('data set and model were loaded successfully!')
	return data_set
