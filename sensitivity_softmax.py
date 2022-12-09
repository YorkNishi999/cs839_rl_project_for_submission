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
ITERATION = 20

# UCB1
outdir = './outputs/sensitivity/softmax.txt'
with open(outdir, 'w') as f:
    pass

anss = []
C_MAX = 10

for cc in range(1, C_MAX, 1):
    ccc = 1 * cc
    
    for i in range(ITERATION):
        command = "python ./gym_adserver/agents/softmax_agent.py --num_ads " + str(ADNUM) + " --impressions " + str(EPOCHS) + " --alpha " + str(ccc)
        print(command)
        os.system(command)

    data = pd.read_csv(outdir, sep=' ', header=None)
    avarage_ucb1 = data.mean()[:EPOCHS]
    ans = avarage_ucb1[EPOCHS-1]
    anss.append(ans)

print(anss)

fig, ax1 = plt.subplots()
ax1.set_ylim([0,0.5])
x = [ i for i in range(1, C_MAX, 1)]
lns2 = ax1.plot(x, anss, color='purple', label='Softmax')

lns = lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('value of beta')
ax1.set_ylabel('Average Reward after 10,000 impressions')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
fig.savefig('./outputs/sensitivity/softmax_sensitivity_' + str(C_MAX) + '.png')