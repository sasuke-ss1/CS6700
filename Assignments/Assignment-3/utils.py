import numpy as np
import gymnasium as gym
from gymnasium import Env
import torch
import random
import matplotlib.pyplot as plt

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

def plot_q_values_best_actions(env, q_values, name, p, d, actions_list=['R', 'G', 'Y', 'B', 'P', 'X']):
    best_actions = np.zeros((5,5))
    placeholder_p = p
    placeholder_d = d
    for row in range(5):
        for col in range(5):
            state = env.encode(row,col,placeholder_p,placeholder_d)
            #x, y , p, d = env.decode(state)
            best_actions[row,col] = np.argmax(q_values[state,:])

    fig, ax = plt.subplots()
    im = ax.imshow(best_actions, cmap='viridis')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks([])
    ax.set_yticks([])


    # Loop over data dimensions and create text annotations
    for i in range(len(best_actions)):
        for j in range(len(best_actions)):
            text = ax.text(j, i, actions_list[int(best_actions[i][j])], ha="center", va="center", color="black", fontsize=20)
    

    ax.set_title(f"Best Actions Grid with p = {placeholder_p}, d = {placeholder_d}")
    plt.savefig(name)


def plot_reward_curves(rewards1, title, xlabel, ylabel, legend, name,rewards2=None, stepsize=5):
    plt.figure(figsize=(10 ,7))
    plt.plot(range(len(rewards1[::stepsize])), rewards1[::stepsize])
    if rewards2:
        plt.plot(range(len(rewards2[::stepsize])), rewards2[::stepsize])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(name)
    
