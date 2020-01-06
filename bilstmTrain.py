import numpy as np
from torch import optim
from part3_parser import Parser
from utils import *
import matplotlib.pyplot as plt


def save_data_to_file(data_name, epochs, loss, acu, choice, with_pretrain=False):
	file_dir = os.path.join(ARTIFACTS_PATH, "{0}_model_result.txt".format(data_name))
	with open(file_dir, "a") as output:
		output.write(
			"Parameters - Choice \'{0}\' Batch size: {1}, epochs: {2}, lr: {3}, embedding length: {4}, lstm hidden dim: {5}\n".format(
				choice, batch_size, epochs, lr, embedding_len, lstm_h_dim))
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


def iterate_model(model, train_data_loader, optimizer, criterion, epoch):
	percentages_show = 5
	limit_to_print = round(len(train_data_loader) * (percentages_show / 100))
	limit_to_print = max(1, limit_to_print)
	for index, batch in enumerate(train_data_loader):
		sentences, tags = batch
		sentences = sentences.cuda()
		tags = tags.cuda()
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


def save_500_acc_dev_to_file(data_name, choice):
	file_dir = os.path.join(ARTIFACTS_PATH, "{0}_dev_500_acc_result.txt".format(data_name))
	with open(file_dir, "a") as output:
		output.write("Data name: {0}, Choice: '{1}'\nDev list: {2}\n\n".format(data_name, choice, str(dev_500_acc)))
	output.close()


def train(model, train_data_loader, dev_data_loader, criterion, optimizer, epochs, data_name):
	dev_acc_list = []
	dev_loss_list = []
	for epoch in range(epochs):
		# train loop
		iterate_model(model, train_data_loader, optimizer, criterion, epoch)

		# calculate performance on dev_data_set
		dev_acc, dev_loss = evaluate_accuracy(model, dev_data_loader, criterion, data_name, epoch)

		dev_acc_list.append(dev_acc)
		dev_loss_list.append(dev_loss)

	print("\n\nTotal Accuracy: " + str(dev_acc_list))
	print("\nTotal Loss: " + str(dev_loss_list))
	save_data_to_file(data_name, epochs, dev_loss_list, dev_acc_list, model.choice, with_pretrain=False)
	save_500_acc_dev_to_file(data_name, model.choice)


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
	limit_to_print = max(1, limit_to_print)
	counter = 0
	avg_acc = 0
	avg_loss = 0
	for index, batch in enumerate(dev_dataset_loader):
		sentences, tags = batch
		sentences = sentences.cuda()
		tags = tags.cuda()
		counter += 1
		y_scores = model(sentences)
		y_hats = torch.argmax(y_scores, dim=1)
		loss = criterion(y_scores, tags)
		current_accuracy = calculate_accuracy(y_hats, tags, data_name)
		avg_acc += current_accuracy
		dev_500_acc.append(current_accuracy)
		avg_loss += float(loss)

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = (index / len(dev_dataset_loader)) * 100
			print("Dev | Epoch: {0} | {1:.2f}% sentences finished".format(epoch + 1, percentages))

	print('\n------ Dev | Finished epoch {0} ------\n'.format(epoch + 1))

	# Calculating acc and loss on all data set.
	acc = (avg_acc / counter) * 100
	loss = avg_loss / counter

	print('***********************************************************************************************')
	print('\nEmbed choice: \'{0}\' Data name:{1} Epoch:{2}, Acc:{3}, Loss:{4}\n'.format(model.choice, data_name,
	                                                                                    epoch + 1,
	                                                                                    acc, loss))
	print('***********************************************************************************************')
	return acc, loss


# Hyper parameters:
# batch_size = 200
# epochs = 50
# lr = 0.001
# embedding_length = 150
# lstm_h_dim = 200
dev_500_acc = []
batch_size = 500
epochs = 1
lr = 0.005
embedding_len = 100
char_embedding_len = 30
lstm_h_dim = 200
choice = 'a'
save_model = True
load_model = True
if __name__ == "__main__":
	# data
	print("before train parser")

	dataTrain = load_dataset(TRAIN_DATASET_DIR) if load_model else Parser("train", "pos")
	print("after train parser and berfore dev parser")
	dataDev = Parser("dev", "pos")
	print("after dev parser")
	dicts = Dictionaries(dataTrain)
	F2I, L2I = dicts.F2I, dicts.L2I
	print("before loaders parser")
	train_loader = make_loader(dataTrain.data, F2I, L2I, batch_size)
	dev_loader = make_loader(dataDev.data, F2I, L2I, batch_size)
	print("after loaders parser")

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
	model = BILSTMNet(vocab_size, embedding_len, lstm_h_dim, output_dim, dicts, char_embedding_len, batch_size, choice)
	if load_model:
		model.load(MODEL_DIR)
	criterion = nn.CrossEntropyLoss(ignore_index=F2I[PAD])
	optimizer = optim.Adam(model.parameters(), lr)
	# train
	train(model, train_loader, dev_loader, criterion, optimizer, epochs, dataTrain.data_name)
	save_model_and_data_sets(model, dataTrain, dataDev)

# model.save("model_" + modelFile + "_" + repr)
#
# qf = open("acc_" + modelFile + "_" + repr, "w")
# qf.write(" ".join(map(str, accs)))
# qf.close()
