import numpy as np
import gymnasium as gym
from gymnasium import Env
import torch
import random

def seed_everything(seed=42) -> None:
    '''
    Seeds torch and numpy rng
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def build_env(name: str, render_mode=None) -> Env:
    '''
    Build the environment
    '''
    return gym.make(name, render_mode=render_mode)

def decode_state(state):
    picDrop, pos = state % 20, state // 20
    x, y = pos % 5, pos // 5
    drop, passenger = picDrop % 4, picDrop //4
    
    return x, y, drop, passenger

def encode_state(x, y, mul=5):
    return mul * x + y

