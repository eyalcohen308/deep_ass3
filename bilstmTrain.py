import torch
import torch.nn as nn
import os
import numpy as np
from torch import optim

from utils import tensorize_sentence
from part3_parser import Parser
import time
from torch.autograd import Variable
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *
import matplotlib.pyplot as plt

PAD_INDEX = 0


class BILSTMNet(nn.Module):
	def __init__(self, vocab_size, embedding_length, lstm_out_dim, output_dim):
		super(BILSTMNet, self).__init__()
		self.batch_size = batch_size
		self.word_emb_dim = embedding_length
		self.hidden_dim = lstm_out_dim

		self.embed = nn.Embedding(vocab_size, embedding_length)

		self.bi_lstm = nn.LSTM(input_size=embedding_length, hidden_size=lstm_out_dim, bidirectional=True, num_layers=2,
		                       batch_first=True)
		self.out = nn.Linear(2 * lstm_out_dim, output_dim)

		self.softmax = nn.LogSoftmax(dim=0)

	def forward(self, sentence):
		# get the len of each vector without padding. if no padding, return len of vector.
		seq_lengths = torch.tensor([get_first_value_index(F2I[PAD], element) for element in sentence])
		total_seq_length = sentence.shape[1]
		embed_sentence = self.embed(sentence)

		packed_x = pack_padded_sequence(embed_sentence, seq_lengths, batch_first=True, enforce_sorted=False)
		packed_lstm_output, _ = self.bi_lstm(packed_x)
		lstm_output, _ = pad_packed_sequence(packed_lstm_output, batch_first=True, padding_value=0,
		                                     total_length=total_seq_length)
		output = self.out(lstm_output)
		output = self.softmax(output)
		output = output.permute(0, 2, 1)
		return output


class Dictionaries:
	def __init__(self, data_set):
		# Word <-> Index:

		extend_vocab = list(data_set.vocab) + [PAD, UNIQUE_WORD]
		extend_tags = list(data_set.tags) + [PAD, UNIQUE_WORD]
		self.F2I = {word: i for i, word in enumerate(extend_vocab)}
		self.I2F = {i: word for i, word in enumerate(extend_vocab)}
		# Label <-> Index:
		self.L2I = {tag: i for i, tag in enumerate(extend_tags)}
		self.I2L = {i: tag for i, tag in enumerate(extend_tags)}

		# char <-> index:
		self.C2I = {char: i for i, char in enumerate(data_set.chars)}
		self.I2C = {i: char for i, char in enumerate(data_set.chars)}

		# pref/suff <-> index:
		self.P2I = {pref: i for i, pref in enumerate(data_set.pref)}
		self.I2P = {i: pref for i, pref in enumerate(data_set.pref)}
		self.S2I = {suff: i for i, suff in enumerate(data_set.suff)}
		self.I2S = {i: suff for i, suff in enumerate(data_set.suff)}


def save_data_to_file(data_name, epochs, loss, acu, with_pretrain=False):
	with open("{0}_model_result.txt".format(data_name), "a") as output:
		output.write(
			"Parameters - Batch size: {0}, epochs: {1}, lr: {2}, embedding length: {3}, lstm hidden dim: {4}\n".format(
				batch_size, epochs, lr, embedding_length, lstm_h_dim))
		output.write(
			"With pre train: {0}, Epochs: {1}\nAccuracy: {2}\nLoss: {3}\n".format(str(with_pretrain), epochs, str(acu),
			                                                                     str(loss)))
	output.close()


def plot_graphs(dev_acc_list, dev_loss_list, epochs, name):
	ticks = int(epochs / 10)
	if not ticks:
		ticks = 1
	plt.plot(range(epochs), dev_acc_list)
	plt.xticks(np.arange(0, epochs, step=1))
	plt.yticks(np.arange(0, 110, step=10))
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.title('{} accuracy'.format(name))
	for i in range(0, len(dev_acc_list), ticks):
		plt.annotate("", (i, dev_acc_list[i]))
	plt.show()

	plt.plot(range(epochs), dev_loss_list)
	plt.xticks(np.arange(0, epochs, step=1))
	plt.yticks(np.arange(0, 4, step=0.5))
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.title('{} loss'.format(name))
	for i in range(0, len(dev_loss_list), ticks):
		plt.annotate("", (i, dev_loss_list[i]))
	plt.show()


def iterate_model(train_data_loader, optimizer, criterion, epoch):
	percentages_show = 5
	limit_to_print = round(len(train_data_loader) * (percentages_show / 100))
	for index, batch in enumerate(train_data_loader):
		sentences, tags = batch

		optimizer.zero_grad()
		output = model(sentences)
		loss = criterion(output, tags)
		loss.backward()
		optimizer.step()

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = (index / len(train_data_loader)) * 100
			print("Train | Epoch: {0} | {1:.2f}% sentences finished".format(epoch + 1, percentages))

	print('\n------ Train | Finished epoch {0} ------\n'.format(epoch + 1))


def train(model, train_data_loader, dev_data_loader, criterion, optimizer, epochs, data_name):
	dev_acc_list = []
	dev_loss_list = []

	for epoch in range(epochs):
		# train loop
		iterate_model(train_data_loader, optimizer, criterion, epoch)

		# calculate performance on dev_data_set
		dev_acc, dev_loss = evaluate_accuracy(model, dev_data_loader, criterion, data_name, epoch)

		dev_acc_list.append(dev_acc)
		dev_loss_list.append(dev_loss)

	save_data_to_file(data_name, epochs, dev_loss_list, dev_acc_list, with_pretrain=False)


# plot_graphs(dev_acc_list, dev_loss_list, epochs, data_name)


def calculate_accuracy(y_hats, tags, data_name):
	good = 0
	bad = 0
	y_hats = y_hats.view(-1)
	tags = tags.view(-1)
	for i in range(len(tags)):
		if tags[i] == L2I[PAD]:
			continue
		if data_name == "ner" and tags[i] == y_hats[i] == L2I['O']:
			continue
		if tags[i] == y_hats[i]:
			good += 1
		else:
			bad += 1

	return good / (good + bad)


def evaluate_accuracy(model, dev_dataset_loader, criterion, data_name, epoch):
	percentages_show = 5
	limit_to_print = round(len(dev_dataset_loader) * (percentages_show / 100))

	counter = 0
	avg_acc = 0
	avg_loss = 0
	for index, batch in enumerate(dev_dataset_loader):
		sentences, tags = batch
		counter += 1
		y_scores = model(sentences)
		y_hats = torch.argmax(y_scores, dim=1)
		loss = criterion(y_scores, tags)

		avg_acc += calculate_accuracy(y_hats, tags, data_name)
		avg_loss += float(loss)

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = (index / len(dev_dataset_loader)) * 100
			print("Dev | Epoch: {0} | {1:.2f}% sentences finished".format(epoch + 1, percentages))

	print('\n------ Dev | Finished epoch {0} ------\n'.format(epoch + 1))

	# Calculating acc and loss on all data set.
	acc = (avg_acc / counter) * 100
	loss = avg_loss / counter

	print('\nData name:{0} Epoch:{1}, Acc:{2}, Loss:{3}'.format(data_name, epoch + 1, acc, loss))
	return acc, loss


# Hyper parameters:
batch_size = 100
epochs = 10
lr = 0.001
embedding_length = 50
lstm_h_dim = 200

if __name__ == "__main__":
	# data
	dataTrain = Parser("train", "pos")
	dataDev = Parser("dev", "pos")
	dicts = Dictionaries(dataTrain)
	F2I, L2I = dicts.F2I, dicts.L2I
	train_loader = make_loader(dataTrain.data, F2I, L2I, batch_size)
	dev_loader = make_loader(dataDev.data, F2I, L2I, batch_size)

	vocab_size = len(F2I)
	output_dim = len(L2I)

	# if sys.argv < 4:
	# 	raise ValueError("invalid inputs")
	#
	# repr, trainFile, modelFile, devFile = sys.argv
	# is_emb = True if repr in {"a", "c", "d"} else False
	# is_sub = True if repr in {"c"} else False
	# is_LSTM = True if repr in {"b", "d"} else False
	# is_pos = True if "pos" in modelFile else False
	#

	# model
	model = BILSTMNet(vocab_size, embedding_length, lstm_h_dim, output_dim)
	criterion = nn.CrossEntropyLoss(ignore_index=F2I[PAD])
	optimizer = optim.Adam(model.parameters(), lr)
	# train
	train(model, train_loader, dev_loader, criterion, optimizer, epochs, dataTrain.data_name)

# model.save("model_" + modelFile + "_" + repr)
#
# qf = open("acc_" + modelFile + "_" + repr, "w")
# qf.write(" ".join(map(str, accs)))
# qf.close()