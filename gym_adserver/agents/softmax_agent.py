import argparse
import sys
import time

import numpy as np
from numpy.random.mtrand import RandomState

import gym
from gym import wrappers, logger

import gym_adserver

class SoftmaxAgent(object):
    def __init__(self, action_space, seed, alpha, max_impressions):
        self.name = "Softmax Agent"
        self.np_random = RandomState(seed)
        self.alpha = alpha
        self.max_impressions = max_impressions
        self.H = np.zeros(action_space.n) + 1/float(action_space.n)
        self.prev_action = None
        self.rewards = []

    def act(self, observation, reward, done):
        ads, current_impressions, _ = observation
        self.rewards.append(reward)
        R_mean = np.mean(self.rewards)
        
        if(self.prev_action == None):
            self.prev_action = np.random.randint(0, len(self.H))
            return self.prev_action

        # update H
        self.H[self.prev_action] += self.alpha * (reward - R_mean) * (1 - self.H[self.prev_action])
        for i in range(len(self.H)):
            if i != self.prev_action:
                self.H[i] -= self.alpha * (reward - R_mean) * self.H[i]
                
        exp_H = np.exp(self.H)
        sum_exp_H = np.sum(exp_H)
        probabilities = exp_H / sum_exp_H

        # Weighted random selection
        self.prev_action = np.random.choice(np.arange(len(ads)), p=probabilities)

        return self.prev_action

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='AdServer-v0')
    parser.add_argument('--num_ads', type=int, default=10)
    parser.add_argument('--impressions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    logger.set_level(logger.INFO)

    time_series_frequency = args.impressions // 10

    # Setup the environment
    env = gym.make(args.env, num_ads=args.num_ads, time_series_frequency=time_series_frequency)
    env.seed(args.seed)

    # Setup the agent
    agent = SoftmaxAgent(env.action_space, args.seed, args.alpha, args.impressions)

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
    
    
    # outdir = './outputs/sensitivity/softmax.txt' # sensitivity test
    outdir = './outputs/softmax.txt' # comparison.py
    with open(outdir, 'a') as f:
        for item in average_reward_list:
            f.write("%s " % item)
        f.write("\n")
    
    print(len(average_reward_list))
    
    # Render the final state and keep the plot window open
    # env.render(freeze=True, output_file=args.output_file)
    
    env.close()