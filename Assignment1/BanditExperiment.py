#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
from unicodedata import name
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 
def run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, param_value, policy='egreedy'):
    avg_r_per_timestep = np.zeros(n_timesteps)
    if policy == 'egreedy':
        for rep in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
            pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
            for timestep in range(n_timesteps):
                a = pi.select_action(epsilon=param_value) # select action
                r = env.act(a) # sample reward
                avg_r_per_timestep[timestep] += (r - avg_r_per_timestep[timestep])/(rep+1)
                pi.update(a,r) # update policy
                # print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    elif policy == 'oi':
        for rep in range(n_repetitions):
            env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
            pi = OIPolicy(n_actions=n_actions, initial_value=param_value) # Initialize policy
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
                a = pi.select_action(c=param_value, t=timestep) # select action
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
    EPSILONS = [0.01,0.05,0.1,0.25]
    all_avg_rewards_egreedy = np.empty((0,n_timesteps), dtype=object)
    epsilon_comparison_plot = ComparisonPlot(title="Comparison of rewards per Epsilon value")
    x=np.arange(n_timesteps)
    for epsilon in EPSILONS:
        avg_rewards_egreedy = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='egreedy',param_value=epsilon)
        epsilon_comparison_plot.add_curve(x,y=smooth(avg_rewards_egreedy,window=smoothing_window),label="Epsilon = %s" % epsilon)
        all_avg_rewards_egreedy = np.append(all_avg_rewards_egreedy, [avg_rewards_egreedy],axis=0)
    epsilon_comparison_plot.save(name="epsilon_comparison.png")

    plot_avg_reward(y=avg_rewards_egreedy, name='egreedy.png')
    
    # Assignment 2: Optimistic init
    INITIAL_VALUES = [0.1,0.5,1.0,2.0]
    all_avg_rewards_oi = np.empty((0,n_timesteps), dtype=object)
    oi_comparison_plot = ComparisonPlot(title="Comparison of rewards per initial value")
    for initial_value in INITIAL_VALUES:
        avg_rewards_oi = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='oi', param_value=initial_value)  
        oi_comparison_plot.add_curve(x,y=smooth(avg_rewards_oi,window=smoothing_window),label="Initial value = %s" % initial_value)
        all_avg_rewards_oi = np.append(all_avg_rewards_oi, [avg_rewards_oi],axis=0)
    oi_comparison_plot.save(name="oi_comparison.png")

    plot_avg_reward(y=avg_rewards_oi, name='oi.png')
    
    
    # Assignment 3: UCB
    C_VALUES = [0.01,0.05,0.1,0.25,0.5,1.0]
    all_avg_rewards_ucb = np.empty((0,n_timesteps), dtype=object)
    ucb_comparison_plot = ComparisonPlot(title="Comparison of rewards per c value")
    for c_value in C_VALUES:
        avg_rewards_ucb = run_repetitions(n_actions, n_timesteps, n_repetitions, smoothing_window, policy='ucb', param_value=c_value)
        ucb_comparison_plot.add_curve(x,y=smooth(avg_rewards_ucb,window=smoothing_window),label="C value = %s" % c_value)
        all_avg_rewards_ucb = np.append(all_avg_rewards_ucb, [avg_rewards_ucb],axis=0)
    ucb_comparison_plot.save(name="ucb_comparison.png")
    plot_avg_reward(y=avg_rewards_ucb, name='ucb.png')

    # Comparison of the three methods
    # comparison_plot = ComparisonPlot(title="Comparison of the three methods")
    # x = np.arange(n_timesteps)
    # comparison_plot.add_curve(x,y=smooth(y=avg_rewards_egreedy,window=smoothing_window),label="e-greedy")
    # comparison_plot.add_curve(x,y=smooth(y=avg_rewards_oi,window=smoothing_window),label="OI")
    # comparison_plot.add_curve(x,y=smooth(y=avg_rewards_ucb,window=smoothing_window),label="UCB")
    # comparison_plot.save(name="comparison.png")

    # This creates the graph similar to the one in the book:
    comparison_plot = ComparisonPlot(title="Comparison of the three methods")
    x = np.arange(n_timesteps)
    comparison_plot.add_curve(x=EPSILONS,y=all_avg_rewards_egreedy.mean(axis=1),label="e-greedy")
    comparison_plot.add_curve(x=INITIAL_VALUES,y=all_avg_rewards_oi.mean(axis=1),label="OI")
    comparison_plot.add_curve(x=C_VALUES,y=all_avg_rewards_ucb.mean(axis=1),label="UCB")
    comparison_plot.save(name="comparison.png")

     
   
if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)