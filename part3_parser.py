import re
from utils import *
import random


class Parser:
	def __init__(self, data_name="pos", data_kind="train", F2I={}, L2I={}):
		file_dir = get_part3_file_directory(data_kind)
		self.data_kind = data_kind
		self.data_name = data_name
		self.dataset = pos_ner_parser(file_dir, data_name, data_kind)
		self.F2I = F2I if F2I else self.generate_f2i()
		if data_kind != "test":
			self.L2I = L2I if L2I else self.generate_l2i()
			self.I2L = {i: l for l, i in self.L2I.items()}

	def generate_f2i(self):
		f2i = {f: i for i, f in
		       enumerate(list(sorted(set([tup[0] for sentence_tup in self.dataset for tup in sentence_tup]))))}

		f2i[UNKNOWN_WORD] = len(f2i)
		return f2i

	def generate_l2i(self):
		l2i = {l: i for i, l in enumerate(list(sorted(set([tup[1] for sentence_tup in self.dataset for tup in sentence_tup]))))}
		l2i[UNKNOWN_WORD] = len(l2i)
		return l2i

	def get_f2i(self):
		return self.F2I

	def get_i2f(self):
		return {i: l for l, i in self.F2I.items()}

	def get_l2i(self):
		return self.L2I

	def get_i2l(self):
		i2l = {i: l for l, i in self.L2I.items()}
		return i2l

	def get_data_set(self, shuffle=True):
		if shuffle:
			random.shuffle(self.dataset)
		return self.dataset


if __name__ == '__main__':
	parser = Parser("pos","train")
	print (parser.dataset)
