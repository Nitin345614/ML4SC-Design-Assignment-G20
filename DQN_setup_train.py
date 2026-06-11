import time
from tqdm import tqdm
import pygame
from pygame import gfxdraw
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import os
from math import cos, sin
import torch
from torch import nn
from copy import deepcopy

import gym_unbalanced_disk

class DQN(nn.Module):
    def __init__(self, env, state_dim=3, action_dim=5):
        super(DQN,self).__init__()
        self.lay1 = nn.Linear(state_dim, 40)  
        self.F1 =  nn.Tanh()  
        self.lay2 = nn.Linear(40, action_dim)  
    
    def forward(self, obs):
        return self.lay2(self.F1(self.lay1(obs)))  


def rollout(Q, env, epsilon=0.1, N_run_timesteps=10_000): 
    #save the following (use .append)
    Start_state = [] #hold an array of (x_t)
    Actions = [] #hold an array of (u_t)
    Rewards = [] #hold an array of (r_{t+1})
    End_state = [] #hold an array of (x_{t+1})
    Terminal = [] #hold an array of (terminal_{t+1})
    # Qfun( a numpy array of the obs) -> a numpy array of Q values
    Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy() 
    with torch.no_grad():
        
        obs, info = env.reset()  
        for i in range(N_run_timesteps):  
            if np.random.uniform()>epsilon:  
                Qnow = Qfun(obs)  
                action = np.argmax(Qnow)  
            else:  
                action = np.random.randint(0, 5)  
            Start_state.append(obs)  
            Actions.append(action)  

            u = [-3,-1.5,0,1.5,3][action]
            obs_next, reward, terminated, truncated, info = env.step(u)  

            Terminal.append(terminated)  
            Rewards.append(reward)  
            End_state.append(obs_next)  

            if terminated or truncated:  
                obs, info = env.reset()  
            else:  
                obs = obs_next  
                
    #error checking:
    assert len(Start_state)==len(Actions)==len(Rewards)==len(End_state)==len(Terminal), f'error in lengths: {len(Start_state)}=={len(Actions)}=={len(Rewards)}=={len(End_state)}=={len(Dones)}'
    return np.array(Start_state), np.array(Actions), np.array(Rewards), np.array(End_state), np.array(Terminal).astype(int)

def eval_Q(Q,env, run_timesteps=200):
    print("Evaluating Q...", end="  ") 
    with torch.no_grad():
        Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
        rewards_acc = 0 
        obs, info = env.reset()  
        for _ in range(run_timesteps):  
            action = np.argmax(Qfun(obs))  
            u = [-3,-1.5,0,1.5,3][action]
            obs, reward, terminated, truncated, info = env.step(u)  
            rewards_acc += reward  
        print(f"[done]")
        return rewards_acc  

def DQN_run_timesteps(Q, optimizer, env, gamma=0.98, use_target_net=False, N_epsilons=21, N_run_timesteps=20000, \
                N_epochs_per_epsilon=10, batch_size=32, N_evals=10, target_net_update_feq=100, epsilon_values=None):
    """
    Trains the Deep Q Network (DQN) using the rollout data.
    """
    best = -float('inf')
    torch.save(Q.state_dict(),' Q-checkpoint-setup-train')
    try:
        if epsilon_values is None:
            epsilon_values = [1.0 - iteration/(N_epsilons-1) for iteration in range(N_epsilons)]

        for iteration, epsilon in enumerate(epsilon_values):
            print(f'Rollout iter={iteration:2d} with epsilon={epsilon:.2%}...')

            #2. rollout
            Start_state, Actions, Rewards, End_state, Terminal = rollout(Q, env, epsilon=epsilon, N_run_timesteps=N_run_timesteps) #e) 2.
            
            #Data conversion, no changes required
            convert = lambda x: [torch.tensor(xi,dtype=torch.float32) for xi in x]
            Start_state, Rewards, End_state, Terminal = convert([Start_state, Rewards, End_state, Terminal])
            Actions = Actions.astype(int)

            t = 0
            for epoch in tqdm(range(N_epochs_per_epsilon), desc='Training Epochs'):
                for i in range(batch_size,len(Start_state)+1,batch_size): 
                    if t%target_net_update_feq==0:
                        Qtarget = deepcopy(Q) #g)
                        pass
                    t += 1
                    
                    Start_state_batch, Actions_batch, Rewards_batch, End_state_batch, Terminal_batch = [d[i-batch_size:i] for d in \
                                                                                                        [Start_state, Actions, Rewards, End_state, Terminal]] #e=) 3.
                    
                    with torch.no_grad(): #3.
                        if use_target_net:
                            pass
                            maxQ = torch.max(Qtarget(End_state_batch),dim=1)[0] #g)
                        else:
                            maxQ = torch.max(Q(End_state_batch),dim=1)[0] #e=) 3.
                    
                    # action_index = np.stack((np.arange(batch_size),Actions_batch),axis=0)
                    # ids = np.arange(batch_size)
                    
                    Qnow = Q(Start_state_batch)
                    # print(f'{action_index.shape=}')
                    # print(f'{Qnow.shape=}')
                    Qnow = Qnow[np.arange(batch_size), Actions_batch] #Q(x_t,u_t) is given
                    # print(Rewards_batch.shape, maxQ.shape, Terminal_batch.shape, Qnow.shape)
                    Loss = torch.mean((Rewards_batch + gamma*maxQ*(1-Terminal_batch) - Qnow)**2) #e) 3.
                    optimizer.zero_grad() #e) 3.
                    Loss.backward() #e) 3.
                    optimizer.step() #e) 3.
                
                print("=================================")
                score = np.mean([eval_Q(Q,env) for i in range(N_evals)])
                print("=================================")
                
                # print(f'iteration={iteration} epoch={epoch} Average Reward per episode:',score)
                if score>best:
                    best = score
                    print('################################# \n new best',best,'saving Q... \n#################################')
                    torch.save(Q.state_dict(),' Q-checkpoint-setup-train')
            
            print('loading best result')
            Q.load_state_dict(torch.load(' Q-checkpoint-setup-train'))
    finally: #this will always run even when using the a KeyBoard Interrupt. 
        print('loading best result')
        Q.load_state_dict(torch.load(' Q-checkpoint-setup-train'))

def show(Q,env,run_timesteps=500):
    with torch.no_grad():
        #you can use Qfun(obs) as a shorthand for the q function.
        Qfun = lambda x: Q(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy() #convert x to torch.tensor -> put in the Q function -> back to numpy
        try:
            obs, info = env.reset() #b)
            Y = [obs]
            for _ in tqdm(range(run_timesteps)): #b)
                action = np.argmax(Qfun(obs)) #b)
                u = [-3,-1.5,0,1.5,3][action]
                obs, reward, terminated, truncated, info = env.step(u) #b)
                Y.append(obs)
        finally: #this will always run even when an error occurs
            env.close()
    
    Y = np.array(Y)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(Y[:,0])
    axs[0].set_ylabel(f'$sin(\\Theta)$')
    axs[1].plot(Y[:,1])
    axs[1].set_ylabel(f'$cos(\\Theta)$')
    axs[2].plot(Y[:,2])
    axs[2].set_ylabel(f'$\\omega$')
    axs[2].grid()
    axs[2].set_xlabel('time step')
    plt.show()


if __name__ == '__main__':
    max_episode_steps = 400
    env = gym_unbalanced_disk.UnbalancedDisk_exp_sincos(umax = 3,dt = 0.025)
    # env = gym.wrappers.TimeLimit(env,max_episode_steps=max_episode_steps)

    gamma = 0.85
    batch_size = 32
    N_epsilons = 3
    N_run_timesteps = 500
    N_epochs_per_epsilon = 3 #f=)
    N_evals = 3 #f=)
    lr = 0.035 #given
    epsilon_values = [0.4, 0.5, 0.6]

    # assert isinstance(env.action_space,gym.spaces.Discrete), 'action space requires to be discrete'
    Q = DQN(env, state_dim=3, action_dim=5)
    Q.load_state_dict(torch.load('Q-checkpoint-setup-train-1st-working'))
    optimizer = torch.optim.Adam(Q.parameters(),lr=lr) #low learning rate
    DQN_run_timesteps(Q, optimizer, env, use_target_net=True, gamma=gamma, N_epsilons=N_epsilons, \
                N_run_timesteps=N_run_timesteps, N_epochs_per_epsilon=N_epochs_per_epsilon, N_evals=N_evals, epsilon_values=epsilon_values)

    show(Q,env, 300)