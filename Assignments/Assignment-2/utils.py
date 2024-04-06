import gymnasium as gym
from collections import deque
import numpy as np
import torch
from gymnasium import Env

def seed_everything(seed=42) -> None:
    '''
    Seeds torch and numpy rng
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)

def build_env(name: str) -> Env:
    '''
    Build the environment
    '''
    return gym.make(name)


class Buffer():
    '''
    The memory buffer for DDQN
    '''
    def __init__(self, size: int) -> None:
        '''
        Initialize the buffer
        '''
        self.size = size
        self.buffer = deque(maxlen=size)
    
    def clear(self):
        '''
        Cleans the buffer
        '''
        self.buffer.clear()

    def space_left(self):
        '''
        Space left in the buffer
        '''
        return self.size - len(self.buffer)

    def __len__(self):
        '''
        Current length of the buffer
        '''
        return len(self.buffer)

    def append(self, exp: tuple[list, int, list, int ,bool]) -> None:
        '''
        Add entry to the buffer
        '''
        self.buffer.append(exp)
    
    def sample(self, batch_size: int, shuffle=True):
        '''
        Sample from the buffer 
        '''
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if shuffle:
            idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in idxs]
        
        else:  
            start = np.random.randint(0, len(self.buffer)-batch_size)
            return [self.buffer[i] for i in range(start, start+batch_size)]
        