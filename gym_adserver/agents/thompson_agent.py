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

class ThompsonAgent(object):
    def __init__(self, action_space, seed, max_impressions):
        self.name = "Thompson Agent"
        self.np_random = RandomState(seed)
        self.max_impressions = max_impressions
        self.prev_action = None
        self.counts = np.zeros(action_space.n)
        self.values = np.zeros(action_space.n)
        self.alpha = np.ones(action_space.n)
        self.beta = np.ones(action_space.n)
        self.ns = np.zeros(action_space.n)
        self.ms = np.zeros(action_space.n)
    
    def select_arm(self):
        return self.beta_sampling(self.alpha, self.beta)

    def act(self, observation, reward, done):
        ads, impressions, _ = observation
        
        # Update the value for the action of the previous act() call
        if self.prev_action != None:     
            self.counts[self.prev_action] = self.counts[self.prev_action] + 1
            n = self.counts[self.prev_action]
            value = self.values[self.prev_action]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            # print("new_value: ", new_value)
            self.values[self.prev_action] = new_value
            
            # update
            self.ns[self.prev_action] += 1
            self.ms[self.prev_action] += reward
            self.alpha[self.prev_action] += self.ms[self.prev_action]
            self.beta[self.prev_action] += self.ns[self.prev_action] - self.ms[self.prev_action]
            
        
        # sampling
        self.prev_action = self.select_arm()
        return self.prev_action

    @staticmethod 
    def beta_sampling(alpha, beta):
        samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(len(alpha))]
        # print("samples: ", samples)
        return np.argmax(samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    # Setup the agent
    agent = ThompsonAgent(env.action_space, args.seed, args.impressions)

    # Simulation loop
    reward = 0
    accumulated_reward_list = []
    average_reward_list = []
    done = False
    observation = env.reset(agent.name)
    for i in range(args.impressions):
        # Action/Feedback
        ad_index = agent.act(observation, reward, done)
        observation, reward, done, _ = env.step(ad_index)
        accumulated_reward_list.append(reward)
        average_reward_list.append(np.mean(accumulated_reward_list))

        
        # Render the current state
        observedImpressions = observation[1]
        # if observedImpressions % time_series_frequency == 0: 
            # env.render()
        
        if done:
            break
    
    # print(average_reward_list)

    outdir = './outputs/thompson.txt' # comparison.py
    with open(outdir, 'a') as f:
        for item in average_reward_list:
            f.write("%s " % item)
        f.write("\n")
    
    print(len(average_reward_list))
    # Render the final state and keep the plot window open
    # env.render(freeze=True, output_file=args.output_file)
    
    env.close()
    
    