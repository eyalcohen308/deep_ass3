import random
import string

import math
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# import torch.optim.lr_scheduler
import torchvision
import torchvision.transforms as transforms

PAD = '<PAD>'
L_PAD = '<L_PAD>'
UNIQUE = 'UUUNKKK'


class MyNet(nn.Module):
    def __init__(self, repr, in_dim, hid_dim1, hid_dim2, hid_dim3, out_dim, I2F=None,
                 pre_trained_embedding=None, letter2I=None, max_word_length=0, letter_embed_dim=0,
                 P2I=None, S2I=None):
        super(MyNet, self).__init__()

        self.repr = repr
        if repr == 'a' or repr == 'd':
            self.word_embedding = nn.Embedding(in_dim, hid_dim1)
            if pre_trained_embedding:
                self.word_embedding.weight.data.copy_(torch.from_numpy(pre_trained_embedding))

        if self.repr == 'b' or repr == 'd':
            self.I2F = I2F
            self.max_word_length = max_word_length
            self.letter2I = letter2I
            self.letter_embed_dim = letter_embed_dim
            self.letter_embedding = nn.Embedding(len(letter2I), letter_embed_dim)
            self.letter_lstm = nn.LSTM(letter_embed_dim, hid_dim1, bidirectional=False, batch_first=True)

        if self.repr == 'c':
            self.I2F = I2F
            self.P2I = P2I
            self.S2I = S2I
            self.word_embedding = nn.Embedding(in_dim, hid_dim1)
            self.prefix_embedding = nn.Embedding(len(P2I), hid_dim1)
            self.suffix_embedding = nn.Embedding(len(S2I), hid_dim1)

        if self.repr == 'd':
            self.linear_layer_cat = nn.Linear(2 * hid_dim1, hid_dim1)

        self.hid_dim1 = hid_dim1
        self.bi_lstm1 = nn.LSTM(hid_dim1, int(hid_dim2 / 2), bidirectional=True, batch_first=True)
        self.bi_lstm2 = nn.LSTM(hid_dim2, int(hid_dim3 / 2), bidirectional=True, batch_first=True)

        self.linear_layer = nn.Linear(hid_dim3, out_dim)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input):
        # input shape is (batch_size, num_sequences)
        seq_lengths = torch.tensor([find_in_tensor(t, F2I[PAD]) for t in input])
        batch_size = len(input)
        total_seq_length = len(input[0])

        if self.repr == 'a':
            input = self.word_embedding(input)
            # input shape is (batch_size, num_sequences,  hid_dim1)

        if self.repr == 'b':
            letter_input, words_lengths = make_letter_input(input, self.I2F, self.max_word_length, self.letter2I)
            # letter_input size is (num_words, max_word_length)
            embed_letter_input = self.letter_embedding(letter_input)
            # embed letter_input size is (num_words, max_word_length, embed_letter_dim)
            packed_letter_input = pack_padded_sequence(embed_letter_input, words_lengths, batch_first=True, enforce_sorted=False)
            _, (final_hidden_state, _) = self.letter_lstm(packed_letter_input)
            # final hidden_state size is (1, num_words, hid_dim1)
            input = final_hidden_state.view(batch_size, total_seq_length,  self.hid_dim1)

        if self.repr == 'c':
            prefix_input, suffix_input = make_prefix_suffix_input(input, self.I2F, self.P2I, self.S2I)
            # prefix_input = suffix_input = input = (batch_size, num_sequences)
            embed_word_input = self.word_embedding(input)
            embed_prefix_input = self.prefix_embedding(prefix_input)
            embed_suffix_input = self.suffix_embedding(suffix_input)

            input = embed_word_input + embed_prefix_input + embed_suffix_input

        if self.repr == 'd':
            embed_output = self.word_embedding(input)
            # embed_input shape is (batch_size, num_sequences,  hid_dim1)
            letter_input, words_lengths = make_letter_input(input, self.I2F, self.max_word_length, self.letter2I)
            # letter_input size is (num_words, max_word_length)
            embed_letter_input = self.letter_embedding(letter_input)
            # embed letter_input size is (num_words, max_word_length, embed_letter_dim)
            packed_letter_input = pack_padded_sequence(embed_letter_input, words_lengths, batch_first=True, enforce_sorted=False)
            _, (final_hidden_state, _) = self.letter_lstm(packed_letter_input)
            # final hidden_state size is (1, num_words, hid_dim1)
            lstm_letter_output = final_hidden_state.view(batch_size, total_seq_length,  self.hid_dim1)
            input = self.linear_layer_cat(torch.cat((embed_output, lstm_letter_output), 2))

        # input shape is (batch_size, num_sequences,  hid_dim1)
        packed_input = pack_padded_sequence(input, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_lstm1_output, _ = self.bi_lstm1(packed_input)
        packed_lstm2_output, _ = self.bi_lstm2(packed_lstm1_output)
        lstm2_output, _ = pad_packed_sequence(packed_lstm2_output, batch_first=True, padding_value=0, total_length=total_seq_length)
        # lstm2_output size is (batch_size, num_sequences, hid_dim3)
        softmax_output = self.softmax(self.linear_layer(lstm2_output))
        # softmax_output size is (batch_size, num_sequences, out_dim)
        output = softmax_output.permute(0, 2, 1)
        # output size is (batch_size, out_dim, num_sequences)
        return output


def make_letter_input(input, I2F, max_word_length, letter2I):
    # input shape is (batch_size, num_sequences)
    word_input = input.view(-1)
    # input shape is (batch_size * num_sequences)
    letter_input = torch.LongTensor(len(word_input), max_word_length)
    words_length = []
    for i, idx in enumerate(word_input):
        word = I2F[int(idx)]
        if word != PAD:
            words_length.append(len(word))
            letter_input[i] = torch.LongTensor(prepare_list(word, max_word_length, letter2I, L_PAD))
        else:
            words_length.append(1)

    return letter_input, words_length


def make_prefix_suffix_input(input, I2F, P2I, S2I):
    # input shape is (batch_size, num_sequences)
    prefix_input = torch.LongTensor(len(input), len(input[0]))
    suffix_input = torch.LongTensor(len(input), len(input[0]))
    for i in range(len(input)):
        for j in range(len(input[0])):
            word = I2F[int(input[i][j])]
            prefix = word[:3]
            suffix = word[-3:]
            if prefix in P2I:
                prefix_input[i][j] = P2I[prefix]
            else:
                prefix_input[i][j] = P2I[UNIQUE[:3]]
            if suffix in S2I:
                suffix_input[i][j] = S2I[suffix]
            else:
                suffix_input[i][j] = S2I[UNIQUE[-3:]]
    return prefix_input, suffix_input


def find_in_tensor(t, idx):
    for i, item in enumerate(t):
        if int(item) == idx:
            return i
    return len(t)


def plot_graphs(dev_acc_list, dev_loss_list, iters, name):
    ticks = int(iters / 10)
    if not ticks:
        ticks = 1
    plt.plot(range(iters + 1), dev_acc_list)
    plt.xticks(np.arange(0, iters + 1, step=1))
    plt.yticks(np.arange(0, 110, step=10))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('{} accuracy'.format(name))
    for i in range(0, len(dev_acc_list), ticks):
        plt.annotate(round(dev_acc_list[i], 1), (i, dev_acc_list[i]))
    plt.show()

    plt.plot(range(iters + 1), dev_loss_list)
    plt.xticks(np.arange(0, iters + 1, step=1))
    plt.yticks(np.arange(0, 4, step=0.5))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('{} loss'.format(name))
    for i in range(0, len(dev_loss_list), ticks):
        plt.annotate(round(dev_loss_list[i], 2), (i, dev_loss_list[i]))
    plt.show()


def train(net, train_data_loader, dev_data_loader, name, iters, lr):
    optimizer = optim.Adam(net.parameters(), lr)
    is_ner = (name.find('ner') != -1)
    dev_acc_list = []
    dev_loss_list = []

    def handle_batch(batch):
        sentences, labels = batch

        optimizer.zero_grad()
        output = net(sentences)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()

    def handle_data(epoch, acc, loss):
        print(epoch, name, 'acc', dev_acc)
        print(epoch, name, 'loss', dev_loss)
        dev_acc_list.append(dev_acc)
        dev_loss_list.append(dev_loss)

    dev_acc, dev_loss = accuracy_n_loss_on_dataset(net, dev_data_loader, is_ner)
    handle_data(0, dev_acc, dev_loss)
    for epoch in range(iters):
        # train loop
        for batch in train_data_loader:
            handle_batch(batch)
        # data
        dev_acc, dev_loss = accuracy_n_loss_on_dataset(net, dev_data_loader, is_ner)
        handle_data(epoch + 1, dev_acc, dev_loss)

    plot_graphs(dev_acc_list, dev_loss_list, iters, name)


def acc_calc(prediction, labels, is_ner):
    good = 0
    bad = 0
    prediction = prediction.view(-1)
    labels = labels.view(-1)
    for i in range(len(labels)):
        if labels[i] == L2I[PAD]:
            continue
        if is_ner and labels[i] == prediction[i] == L2I['O']:
            continue
        if labels[i] == prediction[i]:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def accuracy_n_loss_on_dataset(net, dataset_loader, is_ner):
    counter = 0
    avg_acc = 0
    avg_loss = 0
    for batch in dataset_loader:
        windows, labels = batch

        output = net(windows)
        # output size is (batch_size, out_dim, num_sequences)

        prediction = torch.argmax(output, dim=1)
        # prediction size is (batch_size, num_sequences) such as labels

        loss = F.cross_entropy(output, labels)
        # print(loss)

        counter += 1
        avg_acc += acc_calc(prediction, labels, is_ner)
        avg_loss += float(loss)

    acc = (avg_acc / counter) * 100
    loss = avg_loss / counter
    return acc, loss


def prepare_list(str_list, max_length, mapper, padding):
    idx_list = []
    for s in str_list:
        if s in mapper:
            idx_list.append(mapper[s])
        else:
            idx_list.append(mapper[UNIQUE])
    while len(idx_list) < max_length:
        idx_list.append(mapper[padding])
    return idx_list


def make_loader(data, F2I, L2I, max_length, batch_size):
    x = torch.LongTensor([prepare_list(sentence, max_length, F2I, PAD) for sentence, labels in data])
    y = torch.LongTensor([prepare_list(labels, max_length, L2I, PAD) for sentence, labels in data])
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def parse_data(file_name, is_test=False):
    data = []
    sentence = []
    labels = []
    with open(file_name, 'r') as file:
        for line in file:
            if line == '\n':
                if len(sentence):
                    if is_test:
                        data.append(sentence)
                    else:
                        data.append((sentence, labels))
                sentence = []
                labels = []
            else:
                if is_test:
                    word = line.strip()
                    sentence.append(word)
                else:
                    word, label = line.strip().split()
                    sentence.append(word)
                    labels.append(label)

    if is_test:
        max_length = max([len(sentence) for sentence in data])
    else:
        max_length = max([len(sentence) for sentence, labels in data])

    if is_test:
        return data, None, None, max_length
    word_set = set()
    label_set = set()
    for sentence, labels in data:
        for word in sentence:
            word_set.add(word)
        for label in labels:
            label_set.add(label)

    F2I = {f: i for i, f in enumerate(list(sorted(word_set)))}
    F2I[UNIQUE] = len(F2I)
    F2I[PAD] = len(F2I)
    # feature strings (bigrams) to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(label_set)))}
    L2I[PAD] = len(L2I)
    return data, F2I, L2I, max_length


def make_letter2i(F2I):
    max_word_length = 0
    letter_set = set()
    letter_set.add(L_PAD)
    for f in F2I.keys():
        if len(f) > max_word_length:
            max_word_length = len(f)
        for c in f:
            letter_set.add(c)
    letter2I = {l: i for i, l in enumerate(list(sorted(letter_set)))}
    return letter2I, max_word_length


def make_p2i_s2i(F2I):
    prefixes = set()
    suffixes = set()
    prefixes.add(UNIQUE[:3])
    suffixes.add(UNIQUE[-3:])
    for f in F2I.keys():
        prefixes.add(f[:3])
        suffixes.add(f[-3:])
    P2I = {p: i for i, p in enumerate(list(sorted(prefixes)))}
    S2I = {s: i for i, s in enumerate(list(sorted(suffixes)))}
    return P2I, S2I


name = 'repr=b, without pre_trained, pos'
repr = name[5]
path = './data/pos'.format(name[-3:])

batch_size = 100
iters = 20
lr = 0.005

train_data, F2I, L2I, train_max_length = parse_data('{}/train'.format(path))
dev_data, _, _, dev_max_length = parse_data('{}/dev'.format(path))
test_data, _, _, test_max_length = parse_data('{}/test'.format(path), is_test=True)
max_length = max([train_max_length, dev_max_length, test_max_length])

with_pre_trained = (name.find(' with ') != -1)
if with_pre_trained:
    vecs = np.loadtxt('../wordVectors.txt')
    vocab_file = open('../vocab.txt', 'r')
    vocab = vocab_file.readlines()
    vocab.append(PAD)
    F2I = {word.strip(): i for i, word in enumerate(vocab)}

train_loader = make_loader(train_data, F2I, L2I, max_length, batch_size)
dev_loader = make_loader(dev_data, F2I, L2I, max_length, batch_size)

in_dim = len(F2I)
hid_dim1 = 50
hid_dim2 = 200
hid_dim3 = 400
out_dim = len(L2I)
letter_embed_dim = 30

I2F = {i: f for f, i in F2I.items()}

if repr == 'a':
    net = MyNet(repr, in_dim, hid_dim1, hid_dim2, hid_dim3, out_dim)
elif repr == 'b' or repr == 'd':
    letter2I, max_word_length = make_letter2i(F2I)
    net = MyNet(repr, in_dim, hid_dim1, hid_dim2, hid_dim3, out_dim, I2F=I2F,
                letter2I=letter2I, max_word_length=max_word_length, letter_embed_dim=letter_embed_dim)
elif repr == 'c':
    P2I, S2I = make_p2i_s2i(F2I)
    net = MyNet(repr, in_dim, hid_dim1, hid_dim2, hid_dim3, out_dim,
                I2F=I2F, P2I=P2I, S2I=S2I)
train(net, train_loader, dev_loader, name, iters, lr)

