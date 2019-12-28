import torch
import torch.nn as nn
import os
import numpy as np
from utils import tensorize_sentence
from parser import Parser
import time


class LSTMNet(nn.Module):
	def __init__(self, vocab_size, mlp_input_dim, mlp_hidden_dim, output_size):
		super(LSTMNet, self).__init__()

		# LSTM layer
		self.lstm = nn.LSTM(vocab_size, mlp_input_dim)

		# MLP layers.
		self.mlp_input_layer = nn.Linear(mlp_input_dim, mlp_hidden_dim)
		self.mlp_hidden_layer = nn.Linear(mlp_hidden_dim, output_size)
		self.softmax = nn.LogSoftmax(dim=0)

	def forward(self, sentence):
		lstm_input = sentence.view(1, len(sentence), -1)
		lstm_input = lstm_input.permute(1, 0, 2)
		lstm_output, (final_hidden_state, final_cell_state) = self.lstm(lstm_input)
		output = final_hidden_state[-1]
		output = self.mlp_input_layer(output)
		output = torch.tanh(output)
		output = self.mlp_hidden_layer(output)
		output = self.softmax(output[0])
		return output


def evaluate_accuracy(model, data, F2I, loss_function, epoch):
	"""
	calculate the model performance by given model, dev set, loss function, and Feature to index dictionary.
	:param model: lstm with mlp with two layers.
	:param loss_function: to calculate the loss with.
	:param dev_data_set: list of (string sequence, int tag)
	:param F2I: feature to index dictionary
	:param epochs: how many epochs to run
	:return: nothing
	"""
	good = 0
	sum_loss = 0
	index = 0
	for sentence, tag in data:
		tensor_sentence = tensorize_sentence(sentence, F2I)
		tensor_tag = torch.tensor(tag)
		output = model(tensor_sentence)
		output = output.view(1, -1)
		loss = loss_function(output, tensor_tag)
		sum_loss += float(loss)
		if int(torch.argmax(output)) == int(tensor_tag[0]):
			good += 1
		if index % 500 == 0 and index != 0:
			percentages = (index / len(data)) * 100
			print("Dev | Epoch: {0} | {1:.2f}% sentences finished".format(epoch + 1, percentages))
		index += 1

	print('\n\n------ Dev | Finished epoch {0} ------'.format(epoch))
	print('\tAcc:', good / len(data))
	print('\tLoss', sum_loss / len(data), "\n")


def train_model(model, loss_function, optimizer, train_data_set, dev_data_set, F2I, epochs):
	"""
	train the model by given dataset,dev set, loss function, and Feature to index dictionary.
	:param model: lstm with mlp with two layers.
	:param loss_function: to calculate the loss with.
	:param optimizer: how to calculate and and derive the gradients
	:param train_data_set: list of (string sequence, int tag)
	:param dev_data_set: list of (string sequence, int tag)
	:param F2I: feature to index dictionary
	:param epochs: how many epochs to run
	:return: nothing
	"""
	for epoch in range(epochs):
		index = 0
		for sentence, tag in train_data_set:
			tensor_sentence = tensorize_sentence(sentence, F2I)
			tensor_tag = torch.tensor(tag)
			optimizer.zero_grad()
			output = model(tensor_sentence)
			output = output.view(1, -1)
			loss = loss_function(output, tensor_tag)
			loss.backward()
			optimizer.step()
			if index % 500 == 0 and index != 0:
				percentages = (index / len(train_data_set)) * 100
				print("Train | Epoch: {0} | {1:.2f}% sentences finished".format(epoch + 1, percentages))
			index += 1
		print('\n------ Train | Finished epoch {0} ------\n'.format(epoch))
		evaluate_accuracy(model, dev_data_set, F2I, loss_function, epoch)


if __name__ == "__main__":
	'''
	Model Parameters
	'''
	mlp_input_dim = 50
	mlp_hidden_dim = 100
	output_size = 2
	epochs = 4
	lr = 0.001

	'''
	Parsing the data, using vocabulary from train for define dev words.
	'''
	train_parser = Parser(data_kind="train")
	F2I = train_parser.get_f2i()
	L2I = train_parser.get_l2i()
	vocab_size = len(F2I)
	dev_parser = Parser("dev", F2I, L2I)
	train_data_set = train_parser.get_data_set()
	dev_data_set = dev_parser.get_data_set()

	model = LSTMNet(vocab_size, mlp_input_dim, mlp_hidden_dim, output_size)
	loss_function = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	train_model(model, loss_function, optimizer, train_data_set, dev_data_set, F2I, epochs)
