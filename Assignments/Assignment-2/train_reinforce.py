from utils import *
import torch
import numpy as np
np.random.seed(42)
torch.manual_seed(42)
from tqdm import tqdm
from model import *
from collections import deque
from torch.optim import Adam
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import wandb
import yaml


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

env = build_env('Acrobot-v1')


# episodes = 10000
# hidden_size = 256 
# actor_lr = 1e-3
# value_function_lr = 1e-4 
# gamma = 0.99 
# reward_scale = 1 

def train(args,env, baseline = True):
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]
    agent = REINFORCE(state_size, action_size, args.hidden_size, args.lr_policy, args.lr_value, args.gamma, baseline, device)
    total_reward = 0
    loop_obj = tqdm(range(args.episodes))
    window = deque(maxlen=100)
    rewards = []
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
            reward_list.append(reward*args.reward_scale)
            if done:
                break
            state = next_state
        
        actor_loss, vf_loss = agent.train(state_list, action_list, reward_list)
        rewards.append(total_reward)
        window.append(total_reward)
        loop_obj.set_postfix_str(f"Reward: {sum(window)/100}")   

    return rewards 

def train_wb():
    run = wandb.init() # Initialize the run
    config = wandb.config # Get the config

    if config.baseline == False:
        name = ''
    else:
        name = 'Baseline'

    wandb.run.name = f'REINFORCE_{name}_g_{config.gamma}_r_{config.reward_scale}_h_{config.hidden_size}_lr_[{config.lr_value},{config.lr_policy}]'

    num_expts = 5
    episodic_reward_avgs = []
    for i in range(num_expts):
        episodic_rewards = train(config,env,baseline=config.baseline)
        episodic_reward_avgs.append(episodic_rewards)

    episodic_reward_avgs = np.mean(np.array(episodic_reward_avgs),axis = 0)

    for rewards in episodic_reward_avgs[::50]:
        wandb.log({
            'Episodic reward': rewards
        })


if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument('--wandb_project', '-wp', default='RL-A2-REINFORCE', type=str, help="The wandb project name where run will be stored")
    parser.add_argument('--wandb', '-wb', default=False, type=bool, help="Run WandB")
    parser.add_argument('--gamma', '-g', default='0.99', type=float, help="Discount Factor")
    parser.add_argument('--reward_scale', '-r', default='1', type=float, help="Reward Scale")
    parser.add_argument('--lr_value', '-lrv', default='1e-4', type=float, help="Learning Rate of Value Network")
    parser.add_argument('--lr_policy', '-lra', default='1e-3', type=float, help="Learning Rate of Policy Network")
    parser.add_argument('--hidden_size', '-hs', default='256', type=int, help="Hidden Size")
    parser.add_argument('--episodes', '-e', default='1000', type=int, help="Epsiodes")
    args = parser.parse_args()


    env = build_env('CartPole-v1')

    if args.wandb == False:
        num_expts = 5
        episodic_rewards_list = []
        episodic_rewards_list_Baseline = []
        for i in range(num_expts):
            print(f"Experiment {i+1}")
            episodic_rewards = train(args,env,baseline = False)
            episodic_rewards_list.append(episodic_rewards)
            episodic_rewards = train(args,env,baseline = True)
            episodic_rewards_list_Baseline.append(episodic_rewards)
        episodic_rewards_avg = np.mean(np.array(episodic_rewards_list), axis=0)
        episodic_rewards_std = np.std(np.array(episodic_rewards_list), axis=0)
        episodic_rewards_baseline_avg = np.mean(np.array(episodic_rewards_list_Baseline), axis=0)
        episodic_rewards_baseline_std = np.std(np.array(episodic_rewards_list_Baseline), axis=0)
        plt.figure(figsize = (10,7))
        plt.plot(episodic_rewards_avg, label = 'w/o Baseline')
        plt.fill_between(range(len(episodic_rewards_avg)),episodic_rewards_avg - episodic_rewards_std,episodic_rewards_avg + episodic_rewards_std, color='lightblue', alpha=0.5, label='Mean ± Std Dev')
        plt.plot(episodic_rewards_baseline_avg, label = 'Baseline')
        plt.fill_between(range(len(episodic_rewards_baseline_avg)),episodic_rewards_baseline_avg - episodic_rewards_baseline_std,episodic_rewards_baseline_avg + episodic_rewards_baseline_std, color='orange', alpha=0.5, label='Mean ± Std Dev')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Episodic rewards')
        plt.show()
    else:
        wandb.login(key="ffbdeb8b8eb61fe76925bb00113546a4c1d0581e")
        with open("./sweep_reinforce.yml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        sweep_agent = wandb.agent(sweep_id, function=train_wb)
        pass




    


