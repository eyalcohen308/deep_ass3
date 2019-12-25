import random
from xeger import Xeger

POSITIVE_SEQUENCE_REGEX = r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
NEGATIVE_SEQUENCE_REGEX = r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'
TRAIN_DIR = "train"
DEV_DIR = "dev"
TEST_DIR = "test"


def generate_sequence(reg_exp):
	limit = 25
	word = Xeger(limit=limit)
	word = word.xeger(reg_exp)
	while len(word) > 100:
		limit = limit - 2
		word = Xeger(limit=limit)
		word = word.xeger(reg_exp)
	return word


def create_examples_file(file_name, regex, num_of_sequences):
	sequences = set()
	with open('./data/{0}'.format(file_name), mode='w') as file:
		while len(sequences) < num_of_sequences:
			current_sequence = generate_sequence(regex)
			if current_sequence not in sequences:
				sequences.add(current_sequence)
				file.write("{0}\n".format(current_sequence))
	file.close()


def create_dataset_file(file_name, positive_regex, negative_regex, num_of_sequences, with_label=True):
	sequences = set()
	with open('./data/{0}'.format(file_name), mode='w') as file:
		while len(sequences) < num_of_sequences:
			label = random.randint(0, 1)
			current_sequence = generate_sequence(positive_regex if label else negative_regex)
			if current_sequence not in sequences:
				sequences.add(current_sequence)
				sequence_with_label = "{0}\t{1}\n".format(current_sequence, label)
				sequence_without_label = "{0}\n".format(current_sequence)
				file.write(sequence_with_label if with_label else sequence_without_label)
		file.close()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Deep ex3")
	parser.add_argument("--examples", help="number of examples in files", type=int, default=500)
	args = parser.parse_args()
	examples_num = args.examples
	create_examples_file('pos_examples', POSITIVE_SEQUENCE_REGEX, examples_num)
	create_examples_file('neg_examples', NEGATIVE_SEQUENCE_REGEX, examples_num)

	params = [POSITIVE_SEQUENCE_REGEX, NEGATIVE_SEQUENCE_REGEX, examples_num]

	create_dataset_file(TRAIN_DIR, *params)
	create_dataset_file(DEV_DIR, *params)
	create_dataset_file(TEST_DIR, *params, with_label=False)
