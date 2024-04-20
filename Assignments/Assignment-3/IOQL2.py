from utils import *
from policy import *
from option import *
from collections import deque
from tqdm import tqdm
seed = 42

seed_everything(seed)
eps_min = 0.1
eps_decay = 0.999
eps = 1
policy = EGreedyPolicy(eps, seed)
gamma = 0.9
alpha = 0.1
Neps = 2000
env = build_env('Taxi-v3')
opt = Option2(env, params=1)
nO = 2
q_values_IOQL = np.zeros((env.observation_space.n, nO))
update_IOQL = np.zeros((env.observation_space.n, nO))

rewards = deque(maxlen=100)
plot_rewards = []
loop_obj = tqdm(range(Neps))

for i in loop_obj:
    state = env.reset()[0]
    done = False
    total_steps, total_rewards = 0, 0

    while not done:
        x, y, p, d = env.decode(state)
        
        option = policy(q_values_IOQL, state)
        eps = max(eps_min, eps_decay * eps)

        optDone = False
        state, opt_total_rewards, steps, done = opt.IOQL(state, done, option, q_values_IOQL, update_IOQL, gamma, alpha, eps_min, eps_decay)
        total_rewards += opt_total_rewards
        total_steps += steps
        if total_steps > 200:
            break

    rewards.append(total_rewards)
    plot_rewards.append(total_rewards)
    loop_obj.set_postfix_str(f'Rewards: {sum(rewards)/len(rewards)}')
        

opt.plot_intra_option_q_table(4, 1, 'IOQL_option_Q2')
plot_reward_curves(plot_rewards, 'IOQL', 'Episodes', 'Rewards', ['ioql_agent'], 'IOQL2_rewards')
plot_q_values_best_actions(env, q_values_IOQL, 'IOQL_Q2', 4, 1, ['L', 'D'])

