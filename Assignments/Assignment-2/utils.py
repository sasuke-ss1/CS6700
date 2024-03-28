import gymnasium as gym
from collections import deque
import numpy as np
import torch


def seed_everything(seed=42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def build_env(name: str):
    return gym.make(name)


class Buffer():
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = deque(maxlen=size)
    
    def clear(self):
        self.buffer.clear()

    def space_left(self):
        return self.size - len(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp: tuple[list, int, list, int ,bool]) -> None:
        self.buffer.append(exp)
    
    def sample(self, batch_size: int, shuffle=True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if shuffle:
            idxs = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in idxs]
        
        else:  
            start = np.random.randint(0, len(self.buffer)-batch_size)
            return [self.buffer[i] for i in range(start, start+batch_size)]
        



