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

nX = 5; nY = 5; nPas = 5; nDrop = 4;nO = 4
q_values_SMDP = np.zeros((nPas*nDrop,nO))
updates_SMDP = np.zeros((nPas*nDrop,nO))
rewards = deque(maxlen=100)
loop_obj = tqdm(range(Neps))

for i in loop_obj:
    state = env.reset()[0]
    
    done = False
    total_rewards = 0
    while not done:
        x, y, p, d = env.decode(state)
        subState = encode_state(p, d, nDrop)
        option = policy(q_values_SMDP, subState)
        eps = max(eps_min, eps_decay * eps)

        reward_bar = 0
        optDone = False
        move = 0
        prev = state

        state, reward_bar, move, total_rewards, done = opt(state, done, option, gamma, alpha, eps_min, eps_decay)
            
        _, _, p, d = env.decode(state)
        subState = encode_state(p, d, nDrop)
        
        _, _, p, d = env.decode(prev)
        subPrev = encode_state(p, d, nDrop)

        q_values_SMDP[subPrev, option] += alpha*(reward_bar + (gamma**move)*np.max(q_values_SMDP[subState, :]) - q_values_SMDP[subPrev, option])
        updates_SMDP[subPrev, option] += 1

        x, y, p, d = env.decode(state)

    rewards.append(total_rewards)
    loop_obj.set_postfix_str(f'Rewards: {sum(rewards)/len(rewards)}')
        