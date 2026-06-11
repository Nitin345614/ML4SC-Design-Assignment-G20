import gym_unbalanced_disk
import gymnasium
import numpy as np
from matplotlib import pyplot as plt
import torch


import time
from tqdm import tqdm
import pygame
from pygame import gfxdraw
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import solve_ivp
from os import path
import os
from math import cos, sin
from torch import nn
from copy import deepcopy
# from UnbalancedDisk import UnbalancedDisk_sincos

from UnbalancedDisk import DQN

def show(Q,env,run_timesteps=500, filename=None):
    with torch.no_grad():
        #you can use Qfun(obs) as a shorthand for the q function.
        Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy() #convert x to torch.tensor -> put in the Q function -> back to numpy
        try:
            obs, info = env.reset()
            Y = [obs]
            U = [0]
            for _ in tqdm(range(run_timesteps)): 
                action = np.argmax(Qfun(obs))
                u = [-3,-1.5,0,1.5,3][action]
                obs, reward, terminated, truncated, info = env.step(u)
                Y.append(obs)
                U.append(u)
        finally: #this will always run even when an error occurs
            env.close()
    
    Y = np.array(Y)
    U = np.array(U)
    if filename:
        np.savez(filename + '_4.npz', Y=Y, U=U) #save the trajectory and control inputs for later use
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))
    axs[0].plot(Y[:,0])
    axs[0].set_ylabel(f'$sin(\\Theta)$')
    axs[1].plot(Y[:,1])
    axs[1].set_ylabel(f'$cos(\\Theta)$')
    axs[2].plot(Y[:,2])
    axs[2].set_ylabel(f'$\\omega$')
    axs[2].grid()
    axs[3].plot(U)
    axs[3].set_ylabel('control input $u$')
    axs[3].set_xlabel('time step')
    plt.show()

if __name__ == "__main__":
    env = gym_unbalanced_disk.UnbalancedDisk_exp_sincos(umax = 3,dt = 0.025)
    Q = DQN(env, state_dim=3, action_dim=5)

    # filename = 'Q-checkpoint-setup-metastable'
    # filename = 'Q-checkpoint-setup-train'
    filename = 'Q-checkpoint-setup-train-1st-working'

    Q.load_state_dict(torch.load(filename))

    run_timesteps = 1000
    show(Q,env, run_timesteps=run_timesteps, filename=filename)

    print('done')

