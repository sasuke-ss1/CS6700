from utils import *
from policy import *
from option import *
from collections import deque
from tqdm import tqdm
from argparse import ArgumentParser
seed = 42
seed_everything(seed) # Seed everything

parser = ArgumentParser()
parser.add_argument('--p_min', '-pm', default=0.01, type=float, help='The minimum value of the policy parameter')
parser.add_argument('--p_decay', '-pd', default=0.99, type=float, help='The decay rate of the policy parameter')
parser.add_argument('--p', '-p', default=0.1, type=float, help='The parameter of the policy')
parser.add_argument('--alpha', '-a', default=0.1, type=float, help='Learning rate for the policy')
parser.add_argument('--num_episodes', '-neps', default=2000, type=int, help='Total number of episodes to run')
parser.add_argument('--steps_per_episode', '-spe', default=200, type=int, help='Maximum number of steps per episode')
parser.add_argument('--pick', '-pick', default=0, type=int, help='Pickup location for plotting')
parser.add_argument('--drop', '-drop', default=1, type=int, help='Drop location for plotting')
parser.add_argument('--policy', '-policy', default='egreedy', type=str, help='Policy for agent')
args = parser.parse_args()

eps_min = args.p_min
eps_decay = args.p_decay
eps = args.p
policy = EGreedyPolicy(eps, seed) if args.policy == 'egreedy' else SoftmaxPolicy(eps, seed) 
gamma = 0.9
alpha = args.alpha
Neps = args.num_episodes
env = build_env('Taxi-v3')
opt = Option(env, policy=args.policy, params=eps)
nO = 4

q_values_SMDP = np.zeros((env.observation_space.n,nO + 2))
updates_SMDP = np.zeros((env.observation_space.n,nO + 2))

rewards = deque(maxlen=100)
plot_rewards = []
loop_obj = tqdm(range(Neps))

for i in loop_obj:
    state = env.reset()[0]
    
    done = False
    total_steps, total_rewards = 0, 0

    while not done:
        option = policy(q_values_SMDP, state) # Select option
        eps = max(eps_min, eps_decay * eps) # Update the policy parameter

        prev = state

        state, reward_bar, move, opt_total_rewards, done = opt(state, done, option, gamma, alpha, eps_min, eps_decay) # Execute the option
        total_rewards += opt_total_rewards    
        
        # Update the Q table
        q_values_SMDP[prev, option] += alpha*(reward_bar + (gamma**move)*np.max(q_values_SMDP[state, :]) - q_values_SMDP[prev, option])
        updates_SMDP[prev, option] += 1
        total_steps += move

        if total_steps > args.steps_per_episode:
            break

    rewards.append(total_rewards)
    plot_rewards.append(total_rewards)
    loop_obj.set_postfix_str(f'Rewards: {sum(rewards)/len(rewards)}')

# plot everything
plot_reward_curves(plot_rewards, 'SMDP', 'Episodes', 'Rewards', ['smdp_agent'], 'SMDP_rewards')
opt.plot_intra_option_q_table('SMDP_option_Q')
plot_q_values_best_actions(env, q_values_SMDP, f'SMDP_Q_{args.pick}_{args.drop}', args.pick, args.drop)