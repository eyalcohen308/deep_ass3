import re
from utils import *
import random


class Parser:
	def __init__(self, data_kind="train", F2I={}, L2I={}):
		file_dir = get_part1_file_directory(data_kind)
		self.data_kind = data_kind
		self.dataset = part1_parser(file_dir)
		self.F2I = F2I if F2I else self.generate_f2i()
		self.I2F = {i: l for l, i in self.F2I.items()}
		if data_kind != "test":
			self.L2I = L2I if L2I else self.generate_l2i()
			self.I2L = {i: l for l, i in self.L2I.items()}

	def generate_f2i(self):
		f2i = {f: i for i, f in
		       enumerate(list(sorted(set([char for sentence in self.dataset for char in sentence[0]]))))}

		f2i[UNKNOWN_CHAR] = len(f2i)
		return f2i

	def generate_l2i(self):
		l2i = {l: i for i, l in enumerate(list(sorted(set([data[1][0] for data in self.dataset]))))}
		l2i[UNKNOWN_CHAR] = len(l2i)
		return l2i

	def get_f2i(self):
		return self.F2I

	def get_i2f(self):
		return self.I2F

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
	print("hello")
