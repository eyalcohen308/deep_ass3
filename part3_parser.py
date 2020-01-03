from utils import *
import sys
import numpy as np
import random
import pickle


class Parser:
	def __init__(self, data_kind="train", data_name="pos", to_lower=False, train_path=""):
		self.data = []
		self.tags = set()
		self.vocab = set()
		self.chars = set()
		self.suff = set()
		self.pref = set()
		self.data_name = data_name
		self.data_kind = data_kind
		self.to_lower = to_lower
		self.train_path = train_path

		self.parse_data()

	def parse_data(self):
		sentence = []
		tags = []
		delimiter = ' ' if self.data_name == "pos" else '\t'
		data_dir = self.train_path if self.train_path else get_part3_file_directory(self.data_name, self.data_kind)
		lines_list = open(data_dir).read().splitlines()
		for line in lines_list:

			# Iterate on each word until the end of sentence:
			if line != "":
				parsed_line = line.split(delimiter)
				if self.data_name != "test":
					word, tag = parsed_line
					if tag not in self.tags:
						self.tags.add(tag)
					sentence.append(word)
					tags.append(tag)
				else:
					# If test, won't add tag.
					word = parsed_line[0]
					sentence.append(word)
				# create vocabulary sets:
				if word not in self.vocab:
					self.vocab.add(word)
				for ch in word:
					self.chars.add(ch)
				if word[:3] not in self.pref:
					self.pref.add(word[:3])
				if word[-3:] not in self.suff:
					self.suff.add(word[-3:])
			else:
				# End of sentence:
				self.data.append((sentence, tags))
				sentence = []
				tags = []


# class biLSTMModel:
#
# 	def __init__(self, dataTrain, repr):
# 		self.repr = repr
#
# 		# creates all dictionaries
# 		self.create_dic(dataTrain)
#
# 		# model parameters
# 		self.model = dy.Model()
#
# 		LAYERS = 1
# 		EMB_DIM = 50
# 		HID_DIM = 70
#
# 		# is_emb:
# 		self.E_v = self.model.add_lookup_parameters((len(dataTrain.vocab) + 1, EMB_DIM))
# 		# is_LSTM:
# 		self.E_c = self.model.add_lookup_parameters((len(dataTrain.chars) + 1, EMB_DIM))
# 		self.cLSTM = dy.LSTMBuilder(LAYERS, EMB_DIM, EMB_DIM, self.model)
# 		# is_sub:
# 		self.E_p = self.model.add_lookup_parameters((len(dataTrain.pref) + 1, EMB_DIM))
# 		self.E_s = self.model.add_lookup_parameters((len(dataTrain.suff) + 1, EMB_DIM))
#
# 		self.LSTMf1 = dy.LSTMBuilder(LAYERS, EMB_DIM, HID_DIM, self.model)
# 		self.LSTMb1 = dy.LSTMBuilder(LAYERS, EMB_DIM, HID_DIM, self.model)
# 		self.LSTMf2 = dy.LSTMBuilder(LAYERS, HID_DIM * 2, HID_DIM, self.model)
# 		self.LSTMb2 = dy.LSTMBuilder(LAYERS, HID_DIM * 2, HID_DIM, self.model)
#
# 		self.linear = self.model.add_parameters((EMB_DIM, EMB_DIM * 2))
# 		self.out = self.model.add_parameters((len(dataTrain.tags), HID_DIM * 2))
#
# 		self.trainer = dy.AdamTrainer(self.model)
#
# 	def create_dic(self, dataTrain):
# 		"""
# 		creates all dictionaries
# 		:param dataTrain: train data
# 		:param is_emb:  ords are embedded
# 		:param is_LSTM: use char LSTM
# 		:param is_sub: use pref and suf
# 		:return: N/A
# 		"""
# 		self.tag_to_idx = {tag: i for i, tag in enumerate(dataTrain.tags)}
# 		self.idx_to_tag = {i: tag for i, tag in enumerate(dataTrain.tags)}
# 		# is_emb:
# 		self.word_to_idx = {word: i for i, word in enumerate(dataTrain.vocab)}
# 		self.idx_to_word = {i: word for i, word in enumerate(dataTrain.vocab)}
# 		# is_LSTM:
# 		self.char_to_idx = {char: i for i, char in enumerate(dataTrain.chars)}
# 		self.idx_to_char = {i: char for i, char in enumerate(dataTrain.chars)}
# 		# is_sub:
# 		self.pref_to_idx = {pref: i for i, pref in enumerate(dataTrain.pref)}
# 		self.idx_to_pref = {i: pref for i, pref in enumerate(dataTrain.pref)}
# 		self.suff_to_idx = {suff: i for i, suff in enumerate(dataTrain.suff)}
# 		self.idx_to_suff = {i: suff for i, suff in enumerate(dataTrain.suff)}
#
# 	def forward(self, words, y=None):
# 		"""
# 		model forward
# 		:param words: words
# 		:param y: tags
# 		:return: loss / tags
# 		"""
# 		dy.renew_cg()
#
# 		x = [self.x_repr(w) for w in words]
#
# 		f1_s = self.LSTMf1.initial_state()
# 		b1_s = self.LSTMb1.initial_state()
# 		f1_outputs = f1_s.transduce(x)
# 		b1_outputs = b1_s.transduce(reversed(x))
# 		bilstm_out1 = [dy.concatenate([f, b]) for f, b in zip(f1_outputs, reversed(b1_outputs))]
#
# 		f2_s = self.LSTMf2.initial_state()
# 		b2_s = self.LSTMb2.initial_state()
# 		f2_outputs = f2_s.transduce(bilstm_out1)
# 		b2_outputs = b2_s.transduce(reversed(bilstm_out1))
# 		bilstm_out2 = [dy.concatenate([f, b]) for f, b in zip(f2_outputs, reversed(b2_outputs))]
#
# 		output = dy.parameter(self.out)
#
# 		if y:
# 			y = [self.tag_to_idx.get(t, len(self.tag_to_idx)) for t in y]
#
# 		errors = []
# 		tags = []
# 		for f_b, t in zip(bilstm_out2, y if y else x):
# 			r_t = output * f_b
#
# 			if y:
# 				error = dy.pickneglogsoftmax(r_t, t)
# 				errors.append(error)
# 			else:
# 				chosen = np.argmax(dy.softmax(r_t).npvalue())
# 				tags.append(self.idx_to_tag[chosen])
#
# 		return dy.esum(errors) if y else tags
#
# 	def accuracy(self, sentences):
# 		"""
# 		calculates accuracy (for NER doesn't consider "O")
# 		:param sentences: input
# 		:return: accuracy
# 		"""
# 		good = total = 0
# 		sentenceWords = [[w for w, t in sentence] for sentence in sentences]
# 		y_pred = [model.forward(s) for s in sentenceWords]
# 		y = [[t for w, t in sentence] for sentence in sentences]
#
# 		for a, b in zip(y_pred, y):
# 			for aa, bb in zip(a, b):
# 				total += 1
# 				if aa == bb:
# 					if aa == "O":
# 						total -= 1
# 					else:
# 						good += 1
# 		return float(good) / total
#
# 	def x_repr(self, word):
# 		"""
# 		encodes x according to repr
# 		:param word: word
# 		:return: encoded word according to repr
# 		"""
# 		x = []
# 		if self.repr == "a":
# 			word_idx = self.word_to_idx.get(word, len(self.word_to_idx))
# 			x = self.E_v[word_idx]
# 		elif self.repr == "b":
# 			chars = []
# 			for c in word:
# 				char_idx = self.char_to_idx.get(c, len(self.char_to_idx))
# 				chars.append(self.E_c[char_idx])
# 			c_out = self.cLSTM.initial_state().transduce(chars)
# 			x = c_out[-1]
# 		elif self.repr == "c":
# 			word_idx = self.word_to_idx.get(word, len(self.word_to_idx))
# 			pref_idx = self.pref_to_idx.get(word[:3], len(self.pref_to_idx))
# 			suf_idx = self.suff_to_idx.get(word[-3:], len(self.suff_to_idx))
# 			x = dy.esum([self.E_v[word_idx], self.E_p[pref_idx], self.E_s[suf_idx]])
# 		elif self.repr == "d":
# 			word_idx = self.word_to_idx.get(word, len(self.word_to_idx))
# 			chars = []
# 			for c in word:
# 				char_idx = self.char_to_idx.get(c, len(self.char_to_idx))
# 				chars.append(self.E_c[char_idx])
# 			c_out = self.cLSTM.initial_state().transduce(chars)
# 			x = dy.tanh(dy.parameter(self.linear) * dy.concatenate([self.E_v[word_idx], c_out[-1]]))
# 		return x
#
# 	def save(self, modelFile):
# 		pickle.dump(dataTrain, open(modelFile + "_dict", "wb"))
# 		self.model.save(modelFile)
#
# 	def load(self, modelFile):
# 		self.model.populate(modelFile)
#

# def train(model, dataTrain, dataDev):
# 	"""
# 	train
# 	:param model: model
# 	:param dataTrain: train data
# 	:param dataDev: dev data (data on which accuracy is calcuteD)
# 	:return: accuracies
# 	"""
# 	EPOCH = 5
# 	ACC_RES = 500
# 	accs = []
# 	for i in range(EPOCH):
# 		random.shuffle(dataTrain.data)
# 		for j, sen in enumerate(dataTrain.data, 1):
# 			words = [w for w, t in sen]
# 			tags = [t for w, t in sen]
# 			if j % ACC_RES == 0:
# 				acc = model.accuracy(dataDev.data)
# 				print(i, j, acc)
# 				accs.append(acc)
# 			loss = model.forward(words, tags)
# 			loss.backward()
# 			model.trainer.update()
#
# 	return accs
#

if __name__ == "__main__":
	import argparse

	#
	# parser = argparse.ArgumentParser(description="Deep ex3")
	# parser.add_argument("--data_name", help="choose pos or ner. default is pos", action='store_true')
	# parser.add_argument("--model", help="choose which model to run (a,b,c,d", type=int, default=100)
	# args = parser.parse_args()
	# # 	repr = sys.argv[1]
	# # 	trainFile = sys.argv[2]
	# # 	modelFile = sys.argv[3]
	# # 	devFile = sys.argv[4]
	# #
	# is_emb = True if args.model in {"a", "c", "d"} else False
	# is_sub = True if args.model in {"c"} else False
	# is_LSTM = True if args.model in {"b", "d"} else False

	# data
	dataTrain = Parser("train", "pos")
	dataDev = Parser("dev", "pos")

# model
# model = biLSTMModel(dataTrain, repr)
#
# # train
# accs = train(model, dataTrain, dataDev)
#
# model.save("model_" + modelFi le + "_" + repr)
#
# qf = open("acc_" + modelFile + "_" + repr, "w")
# qf.write(" ".join(map(str, accs)))
# qf.close()
