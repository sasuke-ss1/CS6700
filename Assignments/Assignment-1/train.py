import numpy as np
from algo import *
from grid_world import GridWorld
from policy import SoftmaxPolicy, EGreedyPolicy
from algo import QLearning,SARSA
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import yaml
import wandb


def train_wb():
    '''
    Helper function for wandb hyperparameter sweep
    '''
    run = wandb.init() # Initialize the run
    config = wandb.config # Get the config

    policy_dict = {
        0: SoftmaxPolicy,
        1: EGreedyPolicy
    }

    params = {
        0: config.tau,
        1: config.epsilon
    }

    algo_dict = {
        0: SARSA,
        1: QLearning
    }

    # Initialize the parameters from config
    algorithm_name = 'Q-Learning' if config.algorithm else 'SARSA' 
    policy_name = 'E-Greedy' if config.policy else 'Softmax'
    param = params[config.policy]
    alpha = config.alpha
    gamma = config.gamma
    policy = policy_dict[config.policy](param)
    algorithm = algo_dict[config.algorithm](env, gamma, policy)

    wandb.run.name = f"run_{algorithm_name}_{policy_name}_{param}_{alpha}_{gamma}" # Initialize run name
    
    episodes = 5000
    num_expts = 5
    reward_avgs, steps_avgs = [], []

    for i in range(num_expts):
        algorithm.reset() # Reset everything
        print("Experiment: %d"%(i+1))
        Q, rewards, steps = algorithm.train(alpha, episodes) # Train the algorithm 
        
        reward_avgs.append(rewards)
        steps_avgs.append(steps)


    reward_avgs = np.array(reward_avgs)
    steps_avgs = np.array(steps_avgs)

    avg_steps_q = np.mean(steps_avgs, axis=0)
    avg_rewards_q = np.mean(reward_avgs, axis=0)

    # Log results
    for a_steps, a_rewards in zip(avg_steps_q[::100], avg_rewards_q[::100]):
        wandb.log({
            'steps': a_steps,
            'reward': a_rewards
        })
    
def train(args):
    '''
    Main train function 
    '''

    # Initialize the parameters
    epsilon = args.epsilon
    alpha = args.alpha
    tau = args.tau
    gamma = args.gamma

    env_name = None
    if args.p == 1 and not args.wind:
        env_name = 'Env1'
    elif args.p == 0.7 and not args.wind:
        env_name = 'Env2'
    else:
        env_name = 'Env3'

    # Helper dictionaries
    policy_dict = {
        0: SoftmaxPolicy,
        1: EGreedyPolicy
    }

    params = {
        0: tau,
        1: epsilon
    }

    algo_dict = {
        0: SARSA,
        1: QLearning
    }

    policy = policy_dict[args.policy](params[args.policy])
    algorithm = algo_dict[args.algorithm](env, gamma, policy, name=env_name + f'_{args.start_x}{args.start_y}')


    # Training
    episodes = args.episodes

    num_expts = 5
    reward_avgs, steps_avgs = [], []

    for i in range(num_expts):
        algorithm.reset() # Reset everything
        print("Experiment: %d"%(i+1))
        Q, rewards, steps = algorithm.train(alpha, episodes) # Train the algorithm
        
        reward_avgs.append(rewards)
        steps_avgs.append(steps)
    

    reward_avgs = np.array(reward_avgs)
    steps_avgs = np.array(steps_avgs)

    # Plot heat maps
    algorithm.plot(np.mean(reward_avgs, axis=0),np.mean(steps_avgs, axis=0))
    avg_steps_q = np.mean(steps_avgs, axis=0)[::100]
    avg_rewards_q = np.mean(reward_avgs, axis=0)[::100]
    std_dev_steps = np.std(steps_avgs, axis=0)[::100]
    std_dev_rewards = np.std(reward_avgs, axis=0)[::100]

    # Plot Steps and Reward plots
    plt.figure(figsize = (10,7))
    plt.plot(avg_steps_q)
    plt.fill_between(range(len(avg_steps_q)), avg_steps_q - std_dev_steps,avg_steps_q + std_dev_steps, color='lightblue', alpha=0.5, label='Mean ± Std Dev')
    plt.xlabel('Episode (1 unit = 100 episodes)')
    plt.ylabel('Number of steps to Goal')
    plt.savefig(f'{env_name}_{args.start_x}{args.start_y}_{"Q" if args.algorithm else "SARSA"}_steps.png')
    plt.show()

    plt.figure(figsize = (10,7))
    plt.plot(avg_rewards_q)
    plt.fill_between(range(len(avg_rewards_q)), avg_rewards_q - std_dev_rewards,avg_rewards_q + std_dev_rewards, color='lightblue', alpha=0.5, label='Mean ± Std Dev')
    plt.xlabel('Episode (1 unit = 100 episodes)')
    plt.ylabel('Total Reward')
    plt.savefig(f'{env_name}_{args.start_x}{args.start_y}_{"Q" if args.algorithm else "SARSA"}_reward.png')
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--wandb_project', '-wp', default='RL-A1', type=str, help="The wandb project name where run will be stored")
    parser.add_argument('--episodes', '-ep', default=10000, type=int, help="The number of episodes to play per experiment")
    parser.add_argument('--wind', '-w', type=bool, default=False, help="Sets the wind in the environment")
    parser.add_argument('--p', '-p', default=1, type=float, help='Good transition probability')
    parser.add_argument('--wandb', '-wb', default=False, type=bool, help="Flag to start wandb sweep over hyperparameters")
    parser.add_argument('--start_x', '-sx', default=3, type=int, help='Starting x location')
    parser.add_argument('--start_y', '-sy', default=6, type=int, help='Starting y location')
    parser.add_argument('--epsilon', '-e', type=float, default=0.4, help="Value of epsilon for epsilon greedy policy")
    parser.add_argument('--alpha', '-a', type=float, default=0.4, help="Value of learning rate")
    parser.add_argument('--tau', '-t', default=0.01, type=float, help="Value of temperature for softmax function")
    parser.add_argument('--gamma', '-g', default=0.9, type=float, help="Value of the discounting factor")
    parser.add_argument('--policy', '-policy', default=0, type=int, help="Enter 0 to select Softmax Policy and 1 for epsilon greedy policy")
    parser.add_argument('--algorithm', '-algo', type=int, default=0, help="Enter 0 for SARSA algorithm and 1 for Q Learning")
    args = parser.parse_args()

    # Define Environment
    num_cols = 10
    num_rows = 10
    obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                            [2,7],[3,1],[3,3],[3,5],[4,3],[4,5],[4,7],
                            [5,3],[5,7],[5,9],[6,3],[6,9],[7,1],[7,6],
                            [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])
    bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])
    restart_states = np.array([[3,7],[8,2]])
    start_state = np.array([[args.start_x, args.start_y]])
    goal_states = np.array([[0,9],[2,2],[8,7]])

    # create model
    gw = GridWorld(num_rows=num_rows,
                num_cols=num_cols,
                start_state=start_state,
                goal_states=goal_states, wind = args.wind)
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-100)
    gw.add_transition_probability(p_good_transition=args.p,
                                bias=0.5)
    env = gw.create_gridworld()

    if args.wandb:
        wandb.login(key="ffbdeb8b8eb61fe76925bb00113546a4c1d0581e")
        with open("/kaggle/input/assgn1/sweep.yml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        sweep_agent = wandb.agent(sweep_id, function=train_wb)

    else:
        train(args)
