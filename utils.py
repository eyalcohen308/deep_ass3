import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import re
import pickle
import torch.nn as nn
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pathlib

UNKNOWN_CHAR = '*'
PAD = "*PAD*"
CHAR_PAD = "*CHAR_PAD*"
UNIQUE_WORD = "UUUKKKK"
ARTIFACTS_PATH = "./artifacts"
MODEL_FILE_NAME = "saved_model.pt"
TRAIN_DATASET_NAME = "train_dataset.pickle"
DEV_DATASET_NAME = "dev_dataset.pickle"
pathlib.Path(ARTIFACTS_PATH).mkdir(parents=True, exist_ok=True)

DEV_500_ACC_PATH = os.path.join(ARTIFACTS_PATH, TRAIN_DATASET_NAME)
TRAIN_DATASET_DIR = os.path.join(ARTIFACTS_PATH, TRAIN_DATASET_NAME)
DEV_DATASET_DIR = os.path.join(ARTIFACTS_PATH, DEV_DATASET_NAME)
MODEL_DIR = os.path.join(ARTIFACTS_PATH, MODEL_FILE_NAME)


class BILSTMNet(nn.Module):
	def __init__(self, vocab_size, embedding_len, lstm_out_dim, output_dim, dicts, char_embedding_len, batch_size,
	             choice="a"):
		super(BILSTMNet, self).__init__()
		self.char_embedding_len = char_embedding_len
		self.batch_size = batch_size
		self.word_embed_dim = embedding_len
		self.hidden_dim = lstm_out_dim
		self.choice = choice
		self.dicts = dicts

		# Representation options before model:
		if choice in ['a', 'c', 'd']:
			self.word_embed = nn.Embedding(vocab_size, embedding_len)

		if choice in ['b', 'd']:
			self.char_embed = nn.Embedding(len(dicts.C2I), self.char_embedding_len)
			self.chars_lstm = nn.LSTM(input_size=self.char_embedding_len, hidden_size=embedding_len, batch_first=True)

		if choice == 'c':
			self.prefix_embed = nn.Embedding(len(dicts.P2I), embedding_len)
			self.suffix_embed = nn.Embedding(len(dicts.S2I), embedding_len)

		if choice == 'd':
			self.concat_linear_layer = nn.Linear(embedding_len * 2, embedding_len)

		# Rest of the model:
		self.bi_lstm = nn.LSTM(input_size=embedding_len, hidden_size=lstm_out_dim, bidirectional=True, num_layers=2,
		                       batch_first=True)
		self.out = nn.Linear(2 * lstm_out_dim, output_dim)
		self.softmax = nn.LogSoftmax(dim=0)

	def embed_lstm_a(self, sentence):
		return self.word_embed(sentence)

	def embed_lstm_b(self, sentence, total_seq_length, batch_size):
		char_input = create_char_input(sentence, self.dicts)
		words_len = torch.tensor([get_size_without_pad(self.dicts.C2I[PAD], word) for word in char_input])
		embed_chars = self.char_embed(char_input)
		# It's chars packing time:
		packed_chars_input = pack_padded_sequence(embed_chars, words_len, batch_first=True,
		                                          enforce_sorted=False)
		_, (lstm_last_h_output, _) = self.chars_lstm(packed_chars_input)
		return lstm_last_h_output.view(batch_size, total_seq_length, self.word_embed_dim)

	def embed_lstm_c(self, sentence):
		prefix_input, suffix_input = make_prefix_suffix_input(sentence, self.dicts)
		embed_word_input = self.word_embed(sentence)
		embed_prefix_input = self.prefix_embed(prefix_input)
		embed_suffix_input = self.suffix_embed(suffix_input)
		return embed_word_input + embed_prefix_input + embed_suffix_input

	def embed_lstm_d(self, sentence, total_seq_length, batch_size):
		lstm_a_output = self.embed_lstm_a(sentence)
		lstm_b_output = self.embed_lstm_b(sentence, total_seq_length, batch_size)
		concat_output = torch.cat((lstm_a_output, lstm_b_output), 2)
		return self.concat_linear_layer(concat_output)

	def forward(self, sentence):
		# batch size for resize the shape at the end.
		batch_size = len(sentence)
		# get the len of each vector without padding. if no padding, return len of vector.
		seq_lens_no_pad = torch.tensor([get_size_without_pad(self.dicts.F2I[PAD], element) for element in sentence])
		total_seq_length = sentence.shape[1]

		if self.choice == 'a':
			embed_input = self.embed_lstm_a(sentence)
		elif self.choice == 'b':
			embed_input = self.embed_lstm_b(sentence, total_seq_length, batch_size)
		elif self.choice == 'c':
			embed_input = self.embed_lstm_c(sentence)
		elif self.choice == 'd':
			embed_input = self.embed_lstm_d(sentence, total_seq_length, batch_size)

		# packing embed_input before model layers.
		packed_x = pack_padded_sequence(embed_input, seq_lens_no_pad, batch_first=True, enforce_sorted=False)
		packed_lstm_output, _ = self.bi_lstm(packed_x)
		lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True, padding_value=0,
		                                     total_length=total_seq_length)

		# Rest of the model calculation
		output = self.out(lstm_output)
		output = self.softmax(output)
		output = output.permute(0, 2, 1)
		return output

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)


class Dictionaries:
	def __init__(self, data_set):
		# Word <-> Index:

		extend_vocab = [PAD, UNIQUE_WORD] + list(data_set.vocab)
		extend_tags = [PAD, UNIQUE_WORD] + list(data_set.tags)
		extend_chars = [PAD, UNIQUE_WORD] + list(data_set.chars)
		extend_prefix = [PAD[:3], UNIQUE_WORD[:3]] + list(data_set.pref)
		extend_suffix = [PAD[-3:], UNIQUE_WORD[-3:]] + list(data_set.suff)

		self.F2I = {word: i for i, word in enumerate(extend_vocab)}
		self.I2F = {i: word for i, word in enumerate(extend_vocab)}
		# Label <-> Index:
		self.L2I = {tag: i for i, tag in enumerate(extend_tags)}
		self.I2L = {i: tag for i, tag in enumerate(extend_tags)}

		# char <-> index:
		self.C2I = {char: i for i, char in enumerate(extend_chars)}
		self.I2C = {i: char for i, char in enumerate(extend_chars)}

		# pref/suff <-> index:
		self.P2I = {pref: i for i, pref in enumerate(extend_prefix)}
		self.I2P = {i: pref for i, pref in enumerate(extend_prefix)}
		self.S2I = {suff: i for i, suff in enumerate(extend_suffix)}
		self.I2S = {i: suff for i, suff in enumerate(extend_suffix)}


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


def make_test_loader(data, F2I, batch_size):
	# split the tupled given data to x and y.
	max_sequence_len = max(len(word) for word in data)
	x = torch.LongTensor([convert_to_padded_indexes(sentence, F2I, max_sequence_len) for sentence in data])

	dataset = TensorDataset(x)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_size_without_pad(value, tensor_arr):
	# get the size witout pad. if it zero convert it to 1 ( The packing doe's not execpt zero as word size)
	return (tensor_arr.tolist().index(value) or 1) if value in tensor_arr else len(tensor_arr)


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


def save_model_and_data_sets(model, train_dataset, model_file_path, train_dataset_save_dir):
	model.save(model_file_path)
	with open(train_dataset_save_dir, 'wb') as file:
		pickle.dump(train_dataset, file, pickle.HIGHEST_PROTOCOL)
	print('\nData sets and model were saved successfully!')


def load_dataset(data_set_dir):
	with open(data_set_dir, 'rb') as file:
		data_set = pickle.load(file)
	print('\nData set was loaded successfully!')
	return data_set
