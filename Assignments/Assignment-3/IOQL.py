from utils import *
from policy import *
from option import *
from collections import deque
from tqdm import tqdm
seed = 42

seed_everything(seed)
eps_min = 0.01
eps_decay = 0.99
eps = 0.1
policy = EGreedyPolicy(eps, seed)
policies = {i: EGreedyPolicy(eps, seed) for i in range(4)}
gamma = 0.9
alpha = 0.1
Neps = 5000
env = build_env('Taxi-v3')
opt = Option(env)
cnt = 0
nO = 4
q_values_IOQL = np.zeros((env.observation_space.n, nO))
update_IOQL = np.zeros((env.observation_space.n, nO))

rewards = deque(maxlen=100)
loop_obj = tqdm(range(Neps))

for i in loop_obj:
    state = env.reset()[0]
    
    done = False
    total_rewards = 0
    while not done:
        x, y, p, d = env.decode(state)
        
        option = policy(q_values_IOQL, state)
        eps = max(eps_min, eps_decay * eps)

        optDone = False
        prev = state

        state, total_rewards, done = opt.IOQL(state, done, option, q_values_IOQL, update_IOQL, gamma, alpha, eps_min, eps_decay)
            

    rewards.append(total_rewards)
    loop_obj.set_postfix_str(f'Rewards: {sum(rewards)/len(rewards)}')
        