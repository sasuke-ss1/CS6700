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
            i: np.zeros((env.observation_space.n//20, env.action_space.n - 2)) for i in range(num_options)
        }

        self.goal_name = {
            0: 'R',
            1: 'G',
            2: 'Y',
            3: 'B'
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
            #if p == optNum:
            #    optAct = 4
            #elif d == optNum:
            #    optAct = 5
            if optNum in [0, 1]:
                optAct = 1
            else:
                optAct = 0

        else:
            optAct = policy(self.Q[optNum], encode_state(x, y))
        
        return optAct, optDone

    def __call__(self, state, done, optNum, gamma, alpha, params_min, params_decay):
        reward_bar, total_rewards = 0, 0
        steps = 0
        optDone = False
        
        if optNum >= 4:
            next_state, reward, done,_, _ = self.env.step(optNum)
            steps += 1
            total_rewards += reward
            reward_bar = gamma * reward_bar + reward
            state = next_state
            
            return state, reward_bar, steps, total_rewards, done

        while not optDone and not done:
            optAct, optDone = self.forward(state, optNum, self.policies[optNum])
            next_state, reward, done,_, _ = self.env.step(optAct)

            x, y, _, _ = self.env.decode(state)
            x1, y1, _, _ = self.env.decode(next_state)
            
            pol = self.policies[optNum]
            curr_param = pol.params
            pol.change_params(max(params_min, params_decay * curr_param))

            reward_bar = gamma * reward_bar + reward
            steps += 1
            total_rewards += reward
            reward_surr = reward

            if optDone:
                reward_surr = 20

            self.Q[optNum][encode_state(x, y), optAct] += alpha*(reward_surr + gamma * np.max(self.Q[optNum][encode_state(x1, y1), :]) - self.Q[optNum][encode_state(x, y), optAct])    
            state = next_state

            if steps > 200:
                done=True
                optDone=True

        return state, reward_bar, steps, total_rewards, done
    
    def IOQL(self, state, done, optNum, Q_IOQL, freq_IOQL, gamma, alpha, params_min, params_decay):
        total_rewards = 0
        optDone = False
        steps = 0
        if optNum >= 4:
            next_state, reward, done,_, _ = self.env.step(optNum)
            total_rewards += reward
            steps += 1

            Q_IOQL[state, optNum] += alpha * (reward - Q_IOQL[state, optNum] + gamma * (np.max(Q_IOQL[next_state, :])))
            freq_IOQL[state, optNum] += 1
            
            state = next_state
 
            return state, total_rewards, steps, done

        while not done and not optDone:
            steps += 1
            optAct, optDone = self.forward(state, optNum, self.policies[optNum])
            next_state, reward, done, _, _ = self.env.step(optAct)

            x, y, _, _ =  self.env.decode(state)
            x1, y1, _, _ = self.env.decode(next_state)

            pol = self.policies[optNum]
            curr_param = pol.params
            pol.change_params(max(params_min, params_decay * curr_param))

            total_rewards += reward
            reward_surr = reward
            if optDone:
                reward_surr = 10
            self.Q[optNum][encode_state(x, y), optAct] += alpha*(reward_surr + gamma * np.max(self.Q[optNum][encode_state(x1, y1), :]) - self.Q[optNum][encode_state(x, y), optAct])    

            for i in range(len(self.policies)):
                tmp_act, tmp_done = self.forward(state, i, self.policies[i])
                if tmp_act == optAct:
                    Q_IOQL[state, i] += alpha * (reward - Q_IOQL[state, i] + gamma * (tmp_done * np.max(Q_IOQL[next_state, :]) + (1- tmp_done) * Q_IOQL[next_state, i]))
                    freq_IOQL[state, i] += 1

            state = next_state

        return state, total_rewards, steps, done
    
    def plot_intra_option_q_table(self, name):
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(15)
        fig.set_figwidth(20)

        for i in range(len(self.Q)):
            fig_row = i // 2 
            fig_col = i % 2
            q_values = self.Q[i]
            best_actions = np.zeros((5,5), dtype=int)
            for state in range(500):
                row,col,_,_ = self.env.decode(state)
                best_actions[row, col] = np.argmax(q_values[encode_state(row,col), :])
            action_symbols = ['↓', '↑', '→' , '←', 'P', 'X']
            action_meanings = ['Move Down', 'Move Up', 'Move Right', 'Move Left', 'Pick Up', 'Drop ']
            for row in range(5):
                for col in range(5):
                    action = best_actions[row, col]
                    action_label = action_symbols[action]
                    if (row, col) == (0, 0):
                        color = 'red'
                    elif (row, col) == (0, 4):
                        color = 'green'
                    elif (row, col) == (4, 0):
                        color = 'yellow'
                    elif (row, col) == (4, 3):
                        color = 'blue'
                    else:
                        color = 'lightsteelblue' 
                    rect = plt.Rectangle((col-0.5, 4-row-0.5), 1, 1, fill=True, color = color, alpha=0.6)
                    axs[fig_row,fig_col].add_patch(rect)
                    action = best_actions[row, col]
                    axs[fig_row,fig_col].text(col, 5 - row - 1, action_label, ha='center', va='center', fontsize=20, color='black')

            axs[fig_row,fig_col].axis(xmin=-0.5,xmax=4.5)
            axs[fig_row,fig_col].axis(ymin=-0.5,ymax=4.5)
            axs[fig_row, fig_col].set_xticks([])
            axs[fig_row, fig_col].set_yticks([])  
            axs[fig_row,fig_col].grid(True)
            axs[fig_row,fig_col].set_title(f'Option {self.goal_name[i]} - Go to {self.goal[i]}')

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=action_symbol + ' - ' + action_meaning,
                                 markersize=10) for action_symbol, action_meaning in zip(action_symbols, action_meanings)]
        fig.legend(handles=legend_elements)
        fig.suptitle('Intra-Option Policies (When Passenger is in Taxi)')
        plt.savefig(name)

class Option2():
    def __init__(self, env: Env, num_options=2, policy='egreedy', params=0.1, seed=42):
        self.env = env
        self.goal = {
            0: (0, 0),
            1: (0, 4),
            2: (4, 0),
            3: (4, 3)
        }

        self.goal_name = {
            0: 'L',
            1: 'D',
        }

        self.revGoal = dict(zip(self.goal.values(), self.goal.keys()))

        self.Q = {
            i: np.zeros((env.observation_space.n, env.action_space.n)) for i in range(num_options)
        }

        self.policies = {
            i: EGreedyPolicy(params, seed) if policy == 'egreedy' else SoftmaxPolicy(params, seed) for i in range(num_options)
        }
 
    def forward(self, state, optNum, policy) -> tuple[int, bool]:
        optDone = False
        x, y, p, d = self.env.decode(state)
        if (x, y) in list(self.goal.values()):
            key = self.revGoal[(x, y)]
            if p == key:
                #print(optNum)
                optAct = policy(self.Q[optNum], state)    
                optDone = True if optAct == 4 else False
            elif d == key:
                #print(optNum)
                optAct = policy(self.Q[optNum], state)    
                optDone = True if optAct == 5 else False
            else:
                optAct = policy(self.Q[optNum], state)    

        else:
            optAct = policy(self.Q[optNum], state)
        
        return optAct, optDone

    def __call__(self, state, done, optNum, gamma, alpha, params_min, params_decay):
        reward_bar, total_rewards = 0, 0
        steps = 0
        optDone = False
        while not optDone and not done:
            optAct, optDone = self.forward(state, optNum, self.policies[optNum])
            next_state, reward, done,_, _ = self.env.step(optAct)

            pol = self.policies[optNum]
            curr_param = pol.params
            pol.change_params(max(params_min, params_decay * curr_param))

            reward_bar = gamma * reward_bar + reward
            steps += 1
            total_rewards += reward
            reward_surr = reward
            
            if optDone and optNum == 0 and optAct == 4:
                reward_surr = 20

            if optNum == 1 and optDone and optAct == 5:
                reward_surr = 20

            if optNum == 0 and optAct == 5:
                reward_surr = -20
                optDone = True

            if optNum == 1 and optAct == 4:
                reward_surr = -20
                optDone = True

            self.Q[optNum][state, optAct] += alpha*(reward_surr + gamma * np.max(self.Q[optNum][next_state, :]) - self.Q[optNum][state, optAct])    
            state = next_state
            
        return state, reward_bar, steps, total_rewards, done
    
    def IOQL(self, state, done, optNum, Q_IOQL, freq_IOQL, gamma, alpha, params_min, params_decay):
        total_rewards = 0
        optDone = False
        steps = 0

        while not done and not optDone:
            steps += 1
            optAct, optDone = self.forward(state, optNum, self.policies[optNum])
            next_state, reward, done, _, _ = self.env.step(optAct)

            pol = self.policies[optNum]
            curr_param = pol.params
            pol.change_params(max(params_min, params_decay * curr_param))

            total_rewards += reward
            reward_surr = reward

            if optDone and optNum == 0 and optAct == 4:
                reward_surr = 20

            if optNum == 1 and optDone and optAct == 5:
                reward_surr = 20

            if optNum == 0 and optAct == 5:
                reward_surr = -20
                optDone = True

            if optNum == 1 and optAct == 4:
                reward_surr = -20
                optDone = True
        
            self.Q[optNum][state, optAct] += alpha*(reward_surr + gamma * np.max(self.Q[optNum][next_state, :]) - self.Q[optNum][state, optAct])    

            for i in range(len(self.policies)):
                tmp_act, tmp_done = self.forward(state, i, self.policies[i])
                if tmp_act == optAct:
                    if optNum == 1 and optAct == 4:
                        tmp_done = True
                    if optNum == 0 and optAct == 5:
                        tmp_done = True

                    Q_IOQL[state, i] += alpha * (reward - Q_IOQL[state, i] + gamma * (tmp_done * np.max(Q_IOQL[next_state, :]) + (1- tmp_done) * Q_IOQL[next_state, i]))
                    freq_IOQL[state, i] += 1

            state = next_state

        return state, total_rewards, steps, done

    def plot_intra_option_q_table(self, p, d, name):
        fig, axs = plt.subplots(1, 2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        for i in range(len(self.Q)):
            fig_col = i % 2
            q_values = self.Q[i]
            best_actions = np.zeros((5,5), dtype=int)
            for row in range(5):
                for col in range(5):
                    best_actions[row, col] = np.argmax(q_values[self.env.encode(row,col,p,d), :])
            action_symbols = ['↓', '↑', '→' , '←', 'P', 'X']
            action_meanings = ['Move Down', 'Move Up', 'Move Right', 'Move Left', 'Pick Up', 'Drop ']
            for row in range(5):
                for col in range(5):
                    action = best_actions[row, col]
                    action_label = action_symbols[action]
                    if (row, col) == (0, 0):
                        color = 'red'
                    elif (row, col) == (0, 4):
                        color = 'green'
                    elif (row, col) == (4, 0):
                        color = 'yellow'
                    elif (row, col) == (4, 3):
                        color = 'blue'
                    else:
                        color = 'lightsteelblue' 
                    rect = plt.Rectangle((col-0.5, 4-row-0.5), 1, 1, fill=True, color = color, alpha=0.6)
                    axs[fig_col].add_patch(rect)
                    action = best_actions[row, col]
                    axs[fig_col].text(col, 5 - row - 1, action_label, ha='center', va='center', fontsize=20, color='black')
            axs[fig_col].axis(xmin=-0.5,xmax=4.5)
            axs[fig_col].axis(ymin=-0.5,ymax=4.5)
            axs[fig_col].set_xticks([])
            axs[fig_col].set_yticks([])  
            axs[fig_col].grid(True)
            axs[fig_col].set_title(f'Option {self.goal_name[i]} - Go to {self.goal[i]}')
    
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=action_symbol + ' - ' + action_meaning,
                                 markersize=10) for action_symbol, action_meaning in zip(action_symbols, action_meanings)]
        fig.legend(handles=legend_elements, loc='upper left')
        fig.suptitle('Intra-Option Policies (When Passenger is in Taxi)')
        plt.savefig(name)
    


