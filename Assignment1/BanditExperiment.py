#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 
def run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window):
    for rep in range(n_repetitions):
        env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
        pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
        for timestep in range(n_timesteps):
            a = pi.select_action(epsilon=0.5) # select action
            r = env.act(a) # sample reward
            pi.update(a,r) # update policy
            # print("Test e-greedy policy with action {}, received reward {}".format(a,r))

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window)
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison

    # print("test")
    
    pass

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)