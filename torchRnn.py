# coding: utf-8

from __future__ import print_function
import pickle
import numpy as np
import torch
import random


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
        return Variable(torch.zeros(batch_size, self.hidden_size))

# CHANGE HERE
n_hidden = 144
rnn = RNN(n_features, n_hidden, n_genres)


########## PREPARE FOR TRAINING ##########
def genreFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    #genre_i = top_i[0][0]
    genre_is = top_i[:,0]
    genres = [all_genres[x] for x in genre_is]
    return genres, genre_is

for i in range(10):
    genres, songs, genre_tensor, song_tensor = randomExample(x_train, y_train)
    print('genre0 =', genres[0], '/ song0 =', songs[0])

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

    # changed here so that the printing loss is an average
    return output, torch.mean(loss.data)

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

def accuracy(output, genre_tensor):
    count = 0
    guesses, guess_is = genreFromOutput(output)
    for i in range(batch_size):
        if genre_tensor[i] == guess_is[i]:
            count += 1
    return float(count)/batch_size

start = time.time()
right_count = 0.0
last_v_loss = float('inf')
for iter in range(1, n_iters + 1):
    categories, lines, category_tensor, line_tensor = randomExample(x_train, y_train)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    iterAcc = accuracy(output, category_tensor)
    right_count += iterAcc
    
    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        correct = '✓' if iterAcc > 0.5 else '✗ (%s)' % category
        v_genres, v_songs, v_genre_tensor, v_song_tensor = randomExample(x_test, y_test)
        v_output, v_loss = train(v_genre_tensor, v_song_tensor)
        v_acc = accuracy(v_output, v_genre_tensor)
        print('%d %.2f%% (%s) %.4f %s / %s %s, validation accuracy %.2f' % (iter, float(right_count) / iter * 100, timeSince(start), loss, lines[0], guesses[0], correct, v_acc))
        if v_loss > last_v_loss:
            print("loss stopped decreasing, from ", last_v_loss, " to ", v_loss)
            break;
        else:
            last_v_loss = v_loss
            n_iters += print_every

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#torch.save(rnn.state_dict(), "./nameModel.m")

