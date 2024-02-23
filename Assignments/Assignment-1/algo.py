import numpy as np
from policy import *
from grid_world import GridWorld
from numpy import ndarray
from utils import plot_Q,seq_to_col_row,row_col_to_seq,plot_state_visits
from tqdm import tqdm
from copy import deepcopy
from policy import Policy

class QLearning(object):
    def __init__(self, env: GridWorld, gamma: float, policy: Policy):
        self.env = deepcopy(env)
        self.gamma = gamma
        self.Q = np.zeros((self.env.num_rows, self.env.num_cols, self.env.num_actions))
        self.policy = policy
        self.state_visits = np.zeros((self.env.num_rows, self.env.num_cols, 1))

    def change_env(self, env: GridWorld):
        self.env = deepcopy(env)

    def reset(self):
        self.env.reset()
        self.Q = np.zeros((self.env.num_rows, self.env.num_cols, self.env.num_actions))

    def train(self, alpha, episodes=10000):
        self.state_visits = np.zeros((self.env.num_rows, self.env.num_cols, 1))
        episode_rewards = np.zeros(episodes)
        steps_to_completion = np.zeros(episodes)
        
        for ep in tqdm(range(episodes)):
            tot_reward, steps = 0, 0
            
            # Reset environment
            state = self.env.reset()
            action = self.policy(self.Q, state)
            done = False
            
            while not done:
                self.state_visits[state[0,0],state[0,1],0] += 1
                state_next, reward, done = self.env.step(state, action)
                #print(state, state_next, action)
                action_next = self.policy(self.Q, state_next)
                self.Q[state[0,0]][state[0,1]][action] = self.Q[state[0,0]][state[0,1]][action] + alpha * (reward + self.gamma*(np.max(self.Q[state_next[0,0]][state_next[0,1]])) - self.Q[state[0,0]][state[0,1]][action]) # update equation
                tot_reward += reward
                steps += 1

                state, action = state_next, action_next
                if done:
                    self.state_visits[state_next[0,0],state_next[0,1],0] += 1
            
            episode_rewards[ep] = tot_reward
            steps_to_completion[ep] = steps

        #self.env.render(True, row_col_to_seq(state))
        

        return self.Q, episode_rewards, steps_to_completion
    

    # Need better plot function
    def plot(self, episode_rewards, steps_to_completion):
        plot_Q(self.Q, message = "Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f"%(np.mean(episode_rewards),
                                                                        np.mean(steps_to_completion),
                                                                        self.Q.max(), self.Q.min()))
        plot_state_visits(self.state_visits, message = "Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f"%(np.mean(episode_rewards),
                                                                        np.mean(steps_to_completion),
                                                                        self.Q.max(), self.Q.min()))


class SARSA(object):
    def __init__(self, env: GridWorld, gamma: float, policy: Policy, print_freq=100):
        self.env = deepcopy(env)
        self.gamma = gamma
        self.Q = np.zeros((self.env.num_rows, self.env.num_cols, self.env.num_actions))
        self.policy = policy
        self.print_freq = print_freq
        self.state_visits = np.zeros((self.env.num_rows, self.env.num_cols, 1))

    def change_env(self, env: GridWorld):
        self.env = deepcopy(env)

    def reset(self):
        self.env.reset()
        self.Q = np.zeros((self.env.num_rows, self.env.num_cols, self.env.num_actions))

    def train(self, alpha, episodes=1000):
        self.state_visits = np.zeros((self.env.num_rows, self.env.num_cols, 1))
        episode_rewards = np.zeros(episodes)
        steps_to_completion = np.zeros(episodes)
        for ep in tqdm(range(episodes)):
            tot_reward, steps = 0, 0
            # Reset environment
            state = self.env.reset()
            action = self.policy(self.Q, state)
            done = False

            while not done:
                self.state_visits[state[0,0],state[0,1],0] += 1
                state_next, reward, done = self.env.step(state, action)
                action_next = self.policy(self.Q, state_next)
                self.Q[state[0,0]][state[0,1]][action] = self.Q[state[0,0]][state[0,1]][action] + alpha * (reward + self.gamma * (self.Q[state_next[0,0]][state_next[0,1]][action_next]) - self.Q[state[0,0]][state[0,1]][action]) # update equation                          
                tot_reward += reward
                steps += 1
                
                state, action = state_next, action_next
                if done:
                    self.state_visits[state_next[0,0],state_next[0,1],0] += 1
                
            # self.state_visits[state[0,0],state[0,1],0] += 1
            episode_rewards[ep] = tot_reward
            steps_to_completion[ep] = steps
        # self.env.render(True, row_col_to_seq(state))
        
        return self.Q, episode_rewards, steps_to_completion
    
    def plot(self, episode_rewards, steps_to_completion):
        plot_Q(self.Q, message = "Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f"%(np.mean(episode_rewards),
                                                                    np.mean(steps_to_completion),
                                                                    self.Q.max(), self.Q.min()))
                                                                    
        plot_state_visits(self.state_visits, message = "Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f"%(np.mean(episode_rewards),
                                                                        np.mean(steps_to_completion),
                                                                        self.Q.max(), self.Q.min()))


  
        
        
