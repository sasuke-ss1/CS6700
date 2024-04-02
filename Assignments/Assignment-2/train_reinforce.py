from utils import *
seed_everything(0)
import torch
import numpy as np
from tqdm import tqdm
from model import *
from collections import deque
from torch.optim import Adam
from argparse import ArgumentParser

class REINFORCE():
    def __init__(self, state_size, action_size, hidden_size, actor_lr, vf_lr, gamma, baseline, device):
        self.action_size = action_size
        self.policy = Policy(state_size, action_size, hidden_size).to(device)
        self.vf = Value(state_size, hidden_size).to(device)
        self.actor_optimizer = Adam(self.policy.parameters(), lr=actor_lr)
        self.vf_optimizer = Adam(self.vf.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.device = device
        self.baseline = baseline
        
    def select_action(self, state):
        with torch.no_grad():
            input_state = torch.FloatTensor(state).to(device)
            action_probs = self.policy(input_state)

            action_probs = action_probs.detach().cpu().numpy()
            action = np.random.choice(np.arange(self.action_size), p=action_probs)
        
        return action

    def train(self, state_list, action_list, reward_list):        
        trajectory_len = len(reward_list)
        return_array = np.zeros((trajectory_len,))
        g_return = 0.0

        for i in range(trajectory_len-1,-1,-1):
            g_return = reward_list[i] + self.gamma * g_return
            return_array[i] = g_return
            
        state_t = torch.FloatTensor(state_list).to(device)
        action_t = torch.LongTensor(action_list).to(device).view(-1,1)
        return_t = torch.FloatTensor(return_array).to(device).view(-1,1)

        vf_t = self.vf(state_t).to(device)
        with torch.no_grad():
            delta_t = return_t - vf_t
        
        # Policy loss
        selected_action_prob = self.policy(state_t).gather(1, action_t)
        
        if self.baseline:
            agent_loss = torch.mean(-torch.log(selected_action_prob) * delta_t)
        else:
            agent_loss = torch.mean(-torch.log(selected_action_prob) * return_t)            
        self.actor_optimizer.zero_grad()
        agent_loss.backward()
        self.actor_optimizer.step() 

        # calculate vf loss
        loss_fn = nn.MSELoss()
        vf_loss = loss_fn(vf_t, return_t)
        if self.baseline:
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step() 
        
        return agent_loss.item(), vf_loss.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = build_env('CartPole-v1')
action_size = env.action_space.n
state_size = env.observation_space.shape[0]

episodes = 5000
hidden_size = 256 
actor_lr = 1e-3
value_function_lr = 1e-4 
gamma = 0.99 
reward_scale = 1 

agent = REINFORCE(state_size, action_size, hidden_size, actor_lr, value_function_lr, gamma, True, device)

total_reward = 0

loop_obj = tqdm(range(episodes))

window = deque(maxlen=100)
for ep in loop_obj:
    state = env.reset()[0]
    state_list, action_list, reward_list = [], [], []
    
    total_reward = 0 
    for _ in range(200):
       
        action = agent.select_action(state)

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward*reward_scale)
        
        
        if sum(window)/100>= 195:
            print("Convereged")
            exit()

        if done:
            break
        state = next_state

    actor_loss, vf_loss = agent.train(state_list, action_list, reward_list)
    
    window.append(total_reward)
    loop_obj.set_postfix_str(f"Reward: {sum(window)/100}")    

