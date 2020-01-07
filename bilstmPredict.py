from torch import optim

from utils import *
from part3_parser import Parser
from bilstmTrain import iterate_model
import sys


def submit_prediction_to_file(sentences, y_hats, dicts, test_file_dir, data_name):
	I2F, I2L = dicts.I2F, dicts.I2L
	separate_token = ' ' if data_name == "pos" else '\t'
	with open(test_file_dir + "?", "w+") as output:
		for x, y_hat in zip(sentences, y_hats):
			for curr_x, curr_y_hat in zip(x, y_hat):
				if I2F[int(curr_x)] == PAD:
					continue
				output.write("{0}{1}{2}\n".format(I2F[int(curr_x)], separate_token, I2L[int(curr_y_hat)]))
			output.write("\n")
	output.close()


def predict_tags(model, test_data_loader, test_file_dir, data_name):
	percentages_show = 5
	limit_to_print = round(len(test_data_loader) * (percentages_show / 100))
	limit_to_print = max(1, limit_to_print)
	counter = 0
	for index, batch in enumerate(test_data_loader):
		sentences = batch[0]
		sentences = sentences
		counter += 1
		y_scores = model(sentences)
		y_hats = torch.argmax(y_scores, dim=1)

		submit_prediction_to_file(sentences, y_hats, model.dicts, test_file_dir, data_name)

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = (index / len(test_data_loader)) * 100
			print("Test | {0:.2f}% sentences finished".format(percentages))

	print('***********************************************************************************************\n')
	print("Test | Finished prediction, prediction file dir: '{0}'\n".format(test_file_dir))
	print('***********************************************************************************************')


def test(model, train_data_loader, test_data_loader, test_file_dir, criterion, optimizer, epochs, data_name):
	for epoch in range(epochs):
		# train loop
		iterate_model(model, train_data_loader, optimizer, criterion, epoch)
	# Predict tags (y_hats)
	predict_tags(model, test_data_loader, test_file_dir, data_name)


batch_size = 500
epochs = 1
lr = 0.005
embedding_len = 300
char_embedding_len = 150
lstm_h_dim = 200
choice = 'a'
save_model = True
load_model = True
test_file_dir = "./data/pos/test"

if __name__ == "__main__":
	# data
	if len(sys.argv) != 6:
		raise ValueError("must get 4 parameters, Please run command: "
		                 "'bilstmTrain.py <a/b/c/d> <saved_model_path>"
		                 " <input_file_to_tag> <train_dataset_save_dir> <pos/ner>'")

	_, choice, model_file_path, test_file_dir, train_dataset_save_dir, data_name = sys.argv

	dataTrain = load_dataset(train_dataset_save_dir)
	dataTest = Parser("test", data_name, dataset_path=test_file_dir)
	dicts = Dictionaries(dataTrain)
	F2I, L2I = dicts.F2I, dicts.L2I
	# print("before loaders parser")
	train_loader = make_loader(dataTrain.data, F2I, L2I, batch_size)
	test_loader = make_test_loader(dataTest.data, F2I, batch_size)
	# print("after loaders parser")

	vocab_size = len(F2I)
	output_dim = len(L2I)

	# model
	model = BILSTMNet(vocab_size, embedding_len, lstm_h_dim, output_dim, dicts, char_embedding_len, batch_size, choice)
	if load_model:
		model.load(model_file_path)
	criterion = nn.CrossEntropyLoss(ignore_index=F2I[PAD])
	optimizer = optim.Adam(model.parameters(), lr)

	# train and test model:
	test(model, train_loader, test_loader, test_file_dir, criterion, optimizer, epochs, dataTrain.data_name)
# save_model_and_data_sets(model, dataTrain, dataDev)
