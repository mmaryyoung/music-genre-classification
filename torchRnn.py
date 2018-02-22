# coding: utf-8

import pickle
import numpy as np
import torch
import random



sourceRoot = "/data/hibbslab/jyang/tzanetakis/ver6.0/"
x_train = pickle.load(open(sourceRoot+"x_train.p", "rb"))
y_train = pickle.load(open(sourceRoot+"y_train.p", "rb"))

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

x_train_shape = list(x_train.size())

# 5 sec out of 30 sec, a sixth
all_genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
n_genres = len(all_genres)
n_features = x_train_shape[2]
sample_length = x_train_shape[1]//6

def randomTrainingExample():
    
    idx = random.randint(0, x_train_shape[0]-1)
    jdx = random.randint(0, x_train_shape[1]-sample_length-1)
    song_tensor = x_train[idx][jdx:jdx+sample_length].unsqueeze(1)
    genre_tensor = y_train[idx]
    song = "song" + idx
    genre = all_genres[torch.nonzero(genre_tensor)[0][0]]
    return genre, song, genre_tensor, song_tensor

########## CREATING THE NETWORK ##########


import torch.nn as nn
from torch.autograd import Variable

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
        return Variable(torch.zeros(1, self.hidden_size))

# CHANGE HERE
n_hidden = 128
rnn = RNN(n_features, n_hidden, n_genres)


########## PREPARE FOR TRAINING ##########
def genreFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    genre_i = top_i[0][0]
    return all_genres[genre_i], genre_i

for i in range(10):
    genre, song, genre_tensor, song_tensor = randomTrainingExample()
    print('genre =', genre, '/ song =', song)

########## TRAINING THE NETWORK ##########
criterion = nn.NLLLoss()
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

########## PLOTTING THE RESULTS ##########

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = genreFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#torch.save(rnn.state_dict(), "./nameModel.m")

