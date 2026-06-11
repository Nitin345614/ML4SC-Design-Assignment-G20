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

from UnbalancedDisk import UnbalancedDisk_sincos, DQN, show

max_episode_steps = 300
env = UnbalancedDisk_sincos(dt=0.025)
env = gym.wrappers.TimeLimit(env,max_episode_steps=max_episode_steps)

Q = DQN(env, state_dim=3, action_dim=5)
# Q.load_state_dict(torch.load('Q-checkpoint-1st-working'))
Q.load_state_dict(torch.load('Q-checkpoint-metastable'))
# Q.load_state_dict(torch.load('Q-checkpoint'))
show(Q, env)