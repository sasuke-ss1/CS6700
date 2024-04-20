'''
Generates comparison plots accross all the methods and options.
'''


from utils import *
from policy import *
from option import *
from tqdm import tqdm

seed = 42
seed_everything(seed)
Neps = 2000

def SMDP():
    run_rewards = []
    for _ in range(5):
        eps_min = 0.01
        eps_decay = 0.99
        eps = 0.1
        policy = EGreedyPolicy(eps, seed)
        gamma = 0.9
        alpha = 0.1
        env = build_env('Taxi-v3')
        opt = Option(env)
        nO = 4

        q_values_SMDP = np.zeros((env.observation_space.n,nO + 2))
        updates_SMDP = np.zeros((env.observation_space.n,nO + 2))
        plot_rewards = []
        loop_obj = tqdm(range(Neps))

        for _ in loop_obj:
            state = env.reset()[0]
            
            done = False
            total_steps, total_rewards = 0, 0

            while not done:
                x, y, p, d = env.decode(state)
                option = policy(q_values_SMDP, state)
                eps = max(eps_min, eps_decay * eps)

                reward_bar = 0
                move = 0
                prev = state

                state, reward_bar, move, opt_total_rewards, done = opt(state, done, option, gamma, alpha, eps_min, eps_decay)
                total_rewards += opt_total_rewards    
                
                q_values_SMDP[prev, option] += alpha*(reward_bar + (gamma**move)*np.max(q_values_SMDP[state, :]) - q_values_SMDP[prev, option])
                updates_SMDP[prev, option] += 1
                total_steps += move
                if total_steps > 200:
                    break

            plot_rewards.append(total_rewards)
        run_rewards.append(plot_rewards[::20])
    
    return np.mean(run_rewards, axis=0)

def IOQL():
    run_rewards = []
    for _ in range(5):
        eps_min = 0.01
        eps_decay = 0.99
        eps = 0.1
        policy = EGreedyPolicy(eps, seed)
        gamma = 0.9
        alpha = 0.1

        env = build_env('Taxi-v3')
        opt = Option(env)
        nO = 4

        q_values_IOQL = np.zeros((env.observation_space.n, nO + 2))
        update_IOQL = np.zeros((env.observation_space.n, nO + 2))

        plot_rewards = []
        loop_obj = tqdm(range(Neps))

        for _ in loop_obj:
            state = env.reset()[0]
            
            done = False
            total_steps, total_rewards = 0, 0
            while not done:
                x, y, p, d = env.decode(state)
                
                option = policy(q_values_IOQL, state)
                eps = max(eps_min, eps_decay * eps)
                
                state, opt_total_rewards, steps, done = opt.IOQL(state, done, option, q_values_IOQL, update_IOQL, gamma, alpha, eps_min, eps_decay)
                total_rewards += opt_total_rewards
                total_steps += steps

                if total_steps > 200:
                    break

            plot_rewards.append(total_rewards)
        run_rewards.append(plot_rewards[::20])
    
    return np.mean(run_rewards, axis=0)

def SMDP2():
    run_rewards = []
    for _ in range(5):
        eps_min = 0.01
        eps_decay = 0.99
        eps = 0.2
        policy = EGreedyPolicy(eps, seed)
        gamma = 0.9
        alpha = 0.1
        env = build_env('Taxi-v3')
        opt = Option2(env)
        nO = 2

        q_values_SMDP = np.zeros((env.observation_space.n,nO))
        updates_SMDP = np.zeros((env.observation_space.n,nO))

        plot_rewards = []
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

            plot_rewards.append(total_rewards)
        run_rewards.append(plot_rewards[::20])

    return np.mean(run_rewards, axis=0)

def IOQL2():
    run_rewards = []
    for _ in range(5):
        eps_min = 0.1
        eps_decay = 0.999
        eps = 1
        policy = EGreedyPolicy(eps, seed)
        gamma = 0.9
        alpha = 0.1
        env = build_env('Taxi-v3')
        opt = Option2(env, params=1)
        nO = 2
        q_values_IOQL = np.zeros((env.observation_space.n, nO))
        update_IOQL = np.zeros((env.observation_space.n, nO))

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

                state, opt_total_rewards, steps, done = opt.IOQL(state, done, option, q_values_IOQL, update_IOQL, gamma, alpha, eps_min, eps_decay)
                total_rewards += opt_total_rewards
                total_steps += steps

                if total_steps > 200:
                    break

            plot_rewards.append(total_rewards)
        run_rewards.append(plot_rewards[::20])

    return np.mean(run_rewards, axis=0)


SMDP_rewards = SMDP()
IOQL_rewards = IOQL()
SMDP2_rewards = SMDP2()
IOQL2_rewards = IOQL2()

plt.figure(figsize=(10, 7))
plt.plot(SMDP_rewards, label='SMDP with normal options')
plt.plot(IOQL_rewards, label='IOQL with normal options')
plt.plot(SMDP2_rewards, label='SMDP with alternate options')
plt.plot(IOQL2_rewards, label='IOQL with alternate options')
plt.title('Comparision of different algorithms')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.legend()
plt.savefig('comparison')


