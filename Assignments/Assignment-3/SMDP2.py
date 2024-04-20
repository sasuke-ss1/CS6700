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
gamma = 0.9
alpha = 0.1
Neps = 2000
env = build_env('Taxi-v3')
opt = Option2(env)
cnt = 0
nO = 2

q_values_SMDP = np.zeros((env.observation_space.n,nO))
updates_SMDP = np.zeros((env.observation_space.n,nO))

rewards, plot_rewards = deque(maxlen=100), []
loop_obj = tqdm(range(Neps))

for i in loop_obj:
    state = env.reset()[0]
    
    done = False
    total_steps, total_rewards = 0, 0

    while not done:
        x, y, p, d = env.decode(state)
        option = policy(q_values_SMDP, state)
        eps = max(eps_min, eps_decay * eps)

        reward_bar = 0
        optDone = False
        move = 0
        prev = state

        state, reward_bar, move, opt_total_rewards, done = opt(state, done, option, gamma, alpha, eps_min, eps_decay)
        total_rewards += opt_total_rewards    
        x, y, p, d = env.decode(state)
        
        x1, y1, p, d = env.decode(prev)

        q_values_SMDP[prev, option] += alpha*(reward_bar + (gamma**move)*np.max(q_values_SMDP[state, :]) - q_values_SMDP[prev, option])
        updates_SMDP[prev, option] += 1
        total_steps += move

        if total_steps > 200:
            break

    rewards.append(total_rewards)
    plot_rewards.append(total_rewards)
    loop_obj.set_postfix_str(f'Rewards: {sum(rewards)/len(rewards)}')

opt.plot_intra_option_q_table(4, 1, 'SMDP_option_Q2')
plot_q_values_best_actions(env, q_values_SMDP, 'SMDP_Q2', 4, 1, ['L', 'D'])
plot_reward_curves(plot_rewards, 'SMDP', 'Episodes', 'Rewards', ['smdp2_agent'], 'SMDP_rewards_2')