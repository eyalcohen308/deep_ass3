import random
from xeger import Xeger

POSITIVE_SEQUENCE_REGEX = r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+'
NEGATIVE_SEQUENCE_REGEX = r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+'
TRAIN_DIR = "train"
DEV_DIR = "dev"
TEST_DIR = "test"


def generate_sequence(reg_exp):
	current_seq = SEQUENCE_CHAR_LIMIT
	word = Xeger(limit=current_seq)
	word = word.xeger(reg_exp)
	while len(word) > SEQUENCE_LEN_LIMIT:
		current_seq = current_seq - 2
		word = Xeger(limit=current_seq)
		word = word.xeger(reg_exp)
	return word


def create_examples_file(file_name, regex, num_of_sequences):
	sequences = set()
	with open('./data/{0}'.format(file_name), mode='w') as file:
		while len(sequences) < num_of_sequences:
			current_sequence = generate_sequence(regex)
			if current_sequence not in sequences:
				if len(sequences) % 500 == 0 and len(sequences) != 0:
					percentages = (len(sequences) / num_of_sequences) * 100
					print("Created {0}% sentences in file {1}".format(percentages, file_name))
				sequences.add(current_sequence)
				file.write("{0}\n".format(current_sequence))

	print("Finished creating {0} file".format(file_name))
	file.close()


def create_dataset_file(file_name, positive_regex, negative_regex, num_of_sequences, with_label=True):
	sequences = set()
	with open('./data/{0}'.format(file_name), mode='w') as file:
		while len(sequences) < num_of_sequences:
			label = random.randint(0, 1)
			current_sequence = generate_sequence(positive_regex if label else negative_regex)
			if current_sequence not in sequences:
				if len(sequences) % 500 == 0 and len(sequences) != 0:
					percentages = (len(sequences) / num_of_sequences) * 100
					print("Created {0}% sentences in file {1}".format(percentages, file_name))
				sequences.add(current_sequence)
				sequence_with_label = "{0}\t{1}\n".format(current_sequence, label)
				sequence_without_label = "{0}\n".format(current_sequence)
				file.write(sequence_with_label if with_label else sequence_without_label)

		print("Finished creating {0} file".format(file_name))
		file.close()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Deep ex3")
	parser.add_argument("--examples", help="if one's want to create examples files", action='store_true')
	parser.add_argument("--data_set", help="if one's want to create dataset files (train, dev, test)",
	                    action='store_true')
	parser.add_argument("--examples_size", help="set the number of examples in files, '--examples' must be specified",
	                    type=int, default=500)
	parser.add_argument("--dataset_size", help="set the number of dataset size, '--data_set' must be specified",
	                    type=int, default=10000)
	parser.add_argument("--seq_size", help="set the max sequence size", type=int, default=100)
	args = parser.parse_args()

	global SEQUENCE_LEN_LIMIT, SEQUENCE_CHAR_LIMIT
	SEQUENCE_LEN_LIMIT = args.seq_size
	SEQUENCE_CHAR_LIMIT = SEQUENCE_LEN_LIMIT // 4

	if args.examples:
		create_examples_file('pos_examples', POSITIVE_SEQUENCE_REGEX, args.examples_size)
		create_examples_file('neg_examples', NEGATIVE_SEQUENCE_REGEX, args.examples_size)

	if args.data_set:
		dataset_size = args.dataset_size
		params = [POSITIVE_SEQUENCE_REGEX, NEGATIVE_SEQUENCE_REGEX, dataset_size]
		create_dataset_file(TRAIN_DIR, *params)
		create_dataset_file(DEV_DIR, *params)
		create_dataset_file(TEST_DIR, *params, with_label=False)
