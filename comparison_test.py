import argparse
import sys
import time
import math

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

import gym_adserver

import gym
import optparse
import sys
import os
import random
import numpy as np
import pandas as pd

if "../" not in sys.path:
  sys.path.append("../")
  

ADNUM = 10
EPOCHS = 10000
ITERATION = 2

# thompson
outdir = './outputs/thompson.txt'
print(outdir)

with open(outdir, 'w') as f:
    pass

for i in range(ITERATION):
    os.system("python ./gym_adserver/agents/thompson_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS))

data = pd.read_csv(outdir, sep=' ', header=None)
avarage_thompson = data.mean()[:EPOCHS]
std_thompson = data.std()[:EPOCHS]

# UCB1
outdir = './outputs/ucb1.txt'
print(outdir)

with open(outdir, 'w') as f:
    pass

for i in range(ITERATION):
    os.system("python ./gym_adserver/agents/ucb1_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS))

data = pd.read_csv(outdir, sep=' ', header=None)

avarage_ucb1 = data.mean()[:EPOCHS]
std_ucb1 = data.std()[:EPOCHS]

# Epsilon Greedy
outdir = './outputs/egreedy.txt'
print(outdir)

with open(outdir, 'w') as f:
    pass

for i in range(ITERATION):
    os.system("python ./gym_adserver/agents/epsilon_greedy_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS))

data = pd.read_csv(outdir, sep=' ', header=None)

avarage_egreedy = data.mean()[:EPOCHS]
std_egreedy = data.std()[:EPOCHS]

# softmax
outdir = './outputs/softmax.txt'
print(outdir)

with open(outdir, 'w') as f:
    pass

for i in range(ITERATION):
    os.system("python ./gym_adserver/agents/softmax_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS))

data = pd.read_csv(outdir, sep=' ', header=None)

avarage_softmax = data.mean()[:EPOCHS]
std_softmax = data.std()[:EPOCHS]

# random
outdir = './outputs/random.txt'
print(outdir)
with open(outdir, 'w') as f:
    pass

for i in range(ITERATION):
    os.system("python ./gym_adserver/agents/random_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS))
data = pd.read_csv(outdir, sep=' ', header=None)

avarage_random = data.mean()[:EPOCHS]
std_random = data.std()[:EPOCHS]

fig, ax1 = plt.subplots()
ax1.set_ylim([0,0.5])
x = [ i for i in range(EPOCHS)]
lns1 = ax1.plot(x, avarage_thompson, color='red', label='Thompson Sampling')
lns2 = ax1.plot(x, avarage_ucb1, color='green', label='UCB1')
lns3 = ax1.plot(x, avarage_egreedy, color='blue', label='Epsilon Greedy')
lns4 = ax1.plot(x, avarage_softmax, color='purple', label='Gradient Bandit')
lns5 = ax1.plot(x, avarage_random, color='black', label='Random')

lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Rewards')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
fig.savefig('./epoch' + str(EPOCHS) + '_arms' + str(ADNUM) + '_iterations' + str(ITERATION) + '_comparison.png')

fig, ax1 = plt.subplots()
ax1.set_ylim([0,0.2])
x = [ i for i in range(EPOCHS)]
lns1 = ax1.plot(x, std_thompson, color='red', linestyle = "dashed", label='Thompson Sampling')
lns2 = ax1.plot(x, std_ucb1, color='green', linestyle = "dashed", label='UCB1')
lns3 = ax1.plot(x, std_egreedy, color='blue', linestyle = "dashed", label='Epsilon Greedy')
lns4 = ax1.plot(x, std_softmax, color='purple', linestyle = "dashed", label='Gradient Bandit')
lns5 = ax1.plot(x, std_random, color='black', linestyle = "dashed", label='Random')

lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Std for Average Rewards')

plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.95)
fig.savefig('./epoch' + str(EPOCHS) + '_arms' + str(ADNUM) + '_iterations' + str(ITERATION) +  '_comparison_std.png')