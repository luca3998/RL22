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
 
def run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='egreedy'):
    avg_r_per_timestep = np.zeros(n_timesteps)
    if policy == 'egreedy':
        for rep in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
            pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
            for timestep in range(n_timesteps):
                a = pi.select_action(epsilon=0.5) # select action
                r = env.act(a) # sample reward
                avg_r_per_timestep[timestep] += (r - avg_r_per_timestep[timestep])/(rep+1)
                pi.update(a,r) # update policy
                # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    elif policy == 'oi':
        for rep in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
            pi = OIPolicy(n_actions=n_actions, initial_value=2.0) # Initialize policy
            for timestep in range(n_timesteps):
                a = pi.select_action() # select action
                r = env.act(a) # sample reward
                avg_r_per_timestep[timestep] += (r - avg_r_per_timestep[timestep])/(rep+1)
                pi.update(a,r) # update policy
    elif policy == 'ucb':
        for rep in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
            pi =UCBPolicy(n_actions=n_actions) # Initialize policy
            for timestep in range(n_timesteps):
                a = pi.select_action(c=0.1, t=timestep) # select action
                r = env.act(a) # sample reward
                avg_r_per_timestep[timestep] += (r - avg_r_per_timestep[timestep])/(rep+1)
                pi.update(a,r) # update policy
    else:
        raise Exception("Policy error, please pass one of the following to the policy argument: 'egreedy', 'oi' or 'ucb' ") 
    return avg_r_per_timestep
    

    
def plot_avg_reward(y, name='untitled.png',smoothing=True, save=True):
    egreedy_plot = LearningCurvePlot(title=name)
    egreedy_plot.add_curve(y)
    if smoothing:
        smoothed_line = smooth(y=y, window=smoothing_window)
        egreedy_plot.add_curve(smoothed_line)
    if save:
        egreedy_plot.save(name='egreedy.png')
    

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    avg_rewards_egreedy = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='egreedy')
    
    plot_avg_reward(y=avg_rewards_egreedy, name='egreedy.png')
    
    # Assignment 2: Optimistic init
    avg_rewards_oi = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='oi')
    
    plot_avg_reward(y=avg_rewards_oi, name='oi.png')
    
    
    # Assignment 3: UCB
    avg_rewards_ucb = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='ucb')
    
    plot_avg_reward(y=avg_rewards_oi, name='ucb.png')
     
   
if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)