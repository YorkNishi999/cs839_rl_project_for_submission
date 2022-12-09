import argparse
import sys
import time
import math

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class UCB1Agent(object):
    def __init__(self, action_space, seed, c, max_impressions):
        self.name = "UCB1 Agent"
        self.counts = np.zeros(action_space.n)
        self.values = np.zeros(action_space.n)
        self.np_random = 25
        self.c = c
        self.max_impressions = max_impressions
        self.prev_action = None

    def act(self, observation, reward, done):
        ads, impressions, _ = observation
        # Update the value for the action of the previous act() call
        if self.prev_action != None:
            self.counts[self.prev_action] += 1
            n = self.counts[self.prev_action]
            value = self.values[self.prev_action]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[self.prev_action] = new_value        
        
        # Test each ad once
        
        for ad in ads:
            if ad.impressions == 0:
                self.prev_action = ads.index(ad)
                return self.prev_action

        # print(self.counts)
        # Compute the UCB values
        ucb_values = [0.0] * len(self.values)
        total_counts = sum(self.counts)
        bonus = self.c * np.sqrt(np.log(np.array(total_counts)) / self.counts)
        ucb_values = self.values + bonus
        self.prev_action = np.argmax(ucb_values)
        
        return self.prev_action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--c', type=float, default=0.06)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    # Setup the agent
    agent = UCB1Agent(env.action_space, args.seed, args.c, args.impressions)

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
        #     env.render()
        
        if done:
            break
    
    # Render the final state and keep the plot window open
    # env.render(freeze=True, output_file=args.output_file)
    
    # outdir = './outputs/sensitivity/ucb1.txt' # for sensitivity test
    outdir = './outputs/ucb1.txt' # for comparison test
    with open(outdir, 'a') as f:
        for item in average_reward_list:
            f.write("%s " % item)
        f.write("\n")
    
    print(len(average_reward_list))
    
    env.close()