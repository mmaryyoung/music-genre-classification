# coding: utf-8

from __future__ import print_function
import pickle
import numpy as np
import torch
import random
import sys

import torch.nn as nn
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker

import time
import math


sourceRoot = "/data/hibbslab/jyang/tzanetakis/ver6.0/"
x_train = pickle.load(open(sourceRoot+"x_train_mel.p", "rb"))
y_train = pickle.load(open(sourceRoot+"y_train_mel.p", "rb"))
x_test = pickle.load(open(sourceRoot+"x_test_mel.p", "rb"))
y_test = pickle.load(open(sourceRoot+"y_test_mel.p", "rb"))


# 5 sec out of 30 sec, a sixth
all_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
n_genres = len(all_genres)
n_features = x_train.shape[2]
sample_length = x_train.shape[1]//6

batch_size = 50

def randomExample(x_target, y_target):
    idx = random.randint(0, x_target.shape[0] - batch_size - 1)
    jdx = random.randint(0, x_target.shape[1] - sample_length - 1)
    song_tensor = np.swapaxes(x_target[idx:idx+batch_size, jdx:jdx+sample_length], 0,1)
    song_tensor = Variable(torch.from_numpy(song_tensor))
    genre_tensor = torch.nonzero(torch.from_numpy(y_target[idx:idx+batch_size]))[:,1]
    songs = ["song" + str(x) for x in range(idx, idx+batch_size)]
    genres = [all_genres[x] for x in genre_tensor]
    #song1 = "song" + str(idx)
    #genre1 = all_genres[genre_tensor[0]]
    genre_tensor = Variable(genre_tensor)
    return genres, songs, genre_tensor, song_tensor

def genreFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    #genre_i = top_i[0][0]
    genre_is = top_i[:,0]
    genres = [all_genres[x] for x in genre_is]
    return genres, genre_is


########## CREATING THE NETWORK ##########

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(batch_size, self.hidden_size))

########## PARAMETERS AND SETUPS##########

n_hidden = 144

n_iters = 100000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

learning_rate = 0.0005
criterion = nn.NLLLoss()

rnn = RNN(n_features, n_hidden, n_genres)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_genres, n_genres)
n_confusion = 10000

########## HELPER FUNCTIONS ##########

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def accuracy(outputs, genre_tensor):
    count = 0
    guesses, guess_is = genreFromOutput(outputs)
    for i in range(batch_size):
        if genre_tensor[i] == guess_is[i]:
            count += 1
    return float(count)/batch_size

def evaluate(song_tensor):
    hidden = rnn.initHidden()

    for i in range(song_tensor.size()[0]):
        outputs, hidden = rnn(song_tensor[i], hidden)

    return outputs

def validate():
    losses = []
    accuracies = []
    for i in range(0, x_test.shape[0] - batch_size, batch_size):
        for j in range(0, x_test.shape[1] - sample_length, sample_length):
            song_tensor = Variable(torch.from_numpy(np.swapaxes(x_test[i:i+batch_size, j:j+sample_length],0,1)))
            genre_tensor = Variable(torch.nonzero(torch.from_numpy(y_test[i:i+batch_size]))[:,1])
            outputs = evaluate(song_tensor)
            losses.append(torch.mean(criterion(outputs, genre_tensor).data))
            accuracies.append(accuracy(outputs, genre_tensor.data))
            guesses, guess_is = genreFromOutput(outputs)
            for k in range(batch_size):
                confusion[genre_tensor.data[k]][guess_is[k]] += 1;
    for i in range(n_genres):
        confusion[i] = confusion[i] / confusion[i].sum()        
    return np.mean(losses), np.mean(accuracies)


def train(genre_tensor, song_tensor):
    outputs  = evaluate(song_tensor)

    loss = criterion(outputs, genre_tensor)
    loss.backward()

    torch.nn.utils.clip_grad_norm(rnn.parameters(), 2)

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    # changed here so that the printing loss is an average
    return outputs, torch.mean(loss.data)

########## ACTUAL TRAINING LOOP ###########

start = time.time()
right_count = 0.0
for iter in range(1, n_iters + 1):
    genres, songs, genre_tensor, song_tensor = randomExample(x_train, y_train)
    outputs, loss = train(genre_tensor, song_tensor)
    current_loss += loss

    iterAcc = accuracy(outputs, genre_tensor.data)
    right_count += iterAcc
    
    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        correct = '✓' if iterAcc > 0.5 else '✗' 
        overall_acc = right_count/iter 
        v_loss, v_acc = validate()
        print('%d %.2f%% %.2f%% (%s) %.4f %s / %s, v_acc %.2f%%, loss %.2f' % (iter, overall_acc*100, iterAcc*100, timeSince(start), loss, songs[0], correct, v_acc*100, v_loss))
        sys.stdout.flush()

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn.state_dict(), "musicModel.m")


########## PLOTTING #############
plt.figure()
plt.plot(all_losses)
plt.savefig('loss_trend.png')


# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_genres, rotation=90)
ax.set_yticklabels([''] + all_genres)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.savefig('grid.png')



