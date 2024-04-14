from utils import *
from gymnasium import Env
from policy import *

class Option():
    def __init__(self, env: Env, num_options=4, policy='egreedy', params=0.1, seed=42):
        self.env = env
        self.goal = {
            0: (0, 0),
            1: (0, 4),
            2: (4, 0),
            3: (4, 3)
        }
        self.Q = {
            i: np.zeros((env.observation_space.n//20, env.action_space.n-2)) for i in range(num_options)
        }

        self.policies = {
            i: EGreedyPolicy(params, seed) if policy == 'egreedy' else SoftmaxPolicy(params, seed) for i in range(num_options)
        }
        

    def forward(self, state, optNum, policy) -> tuple[int, bool]:
        assert optNum < 4, 'Only 4 options Implemented' 
        
        optDone = False
        x, y, p, d = self.env.decode(state)

        if (x, y) == self.goal[optNum]:
            optDone = True

            if p == optNum:
                optAct = 4
            elif d == optNum:
                optAct = 5
            elif optNum in [0, 1]:
                optAct = 1
            else:
                optAct = 0

        else:
            optAct = policy(self.Q[optNum], encode_state(x, y))
        
        return optAct, optDone

    def __call__(self, state, done, optNum, gamma, alpha, param_min, params_decay):
        reward_bar, total_rewards = 0, 0
        steps = 0
        optDone = False
        while not optDone and not done:
            optAct, optDone = self.forward(state, optNum, self.policies[optNum])
            x, y, _, _ = self.env.decode(state)
            next_state, reward, done,_, _ = self.env.step(optAct)
            x1, y1, _, _ = self.env.decode(next_state)
            reward_bar = gamma * reward_bar + reward
            steps += 1
            
            for pol in self.policies.values():
                curr_param = pol.params
                pol.change_params(max(param_min, params_decay * curr_param))
            
            total_rewards += reward
        
            if optAct < 4:
                self.Q[optNum][encode_state(x, y), optAct] += alpha*(reward + gamma * np.max(self.Q[optNum][encode_state(x1, y1), :]) - self.Q[optNum][encode_state(x, y), optAct])
            
            state = next_state

        return state, reward_bar, steps, total_rewards, done