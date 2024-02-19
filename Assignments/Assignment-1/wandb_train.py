import wandb
from algo import *
from grid_world import GridWorld
import numpy as np
from policy import SoftmaxPolicy, EGreedyPolicy
from algo import QLearning,SARSA
import matplotlib.pyplot as plt


wandb.init()


sweep_config = {
    "name": "Hyperparameter_Tuning",
    "method": "grid",
    "parameters": {
        "beta": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        "alpha": {"values": [0.1, 0.2, 0.3, 0.4]},
        "gamma": {"values": [0.8, 0.9]},
        'policy':{'values': ['Softmax', "E-greedy"]},
        'algorithm':{'values': ['Q-Learning', "SARSA"]}
    },
}


sweep_id = wandb.sweep(sweep_config)


def train():    
    num_cols = 10
    num_rows = 10
    obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                            [2,7],[3,1],[3,3],[3,5],[4,3],[4,5],[4,7],
                            [5,3],[5,7],[5,9],[6,3],[6,9],[7,1],[7,6],
                            [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])
    bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[9,9]])
    restart_states = np.array([[3,7],[8,2]])
    start_state = np.array([[0, 4]])
    goal_states = np.array([[0,9],[2,2],[8,7]])

    # create model
    gw = GridWorld(num_rows=num_rows,
                num_cols=num_cols,
                start_state=start_state,
                goal_states=goal_states, wind = False)
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    gw.add_rewards(step_reward=-1,
                goal_reward=10,
                bad_state_reward=-6,
                restart_state_reward=-100)
    gw.add_transition_probability(p_good_transition=1,
                                bias=0.5)
    env = gw.create_gridworld()

    beta = wandb.config.beta
    alpha = wandb.config.alpha
    gamma = wandb.config.gamma
    
    if wandb.config.policy == 'Softmax':
        policy = SoftmaxPolicy(beta)
    else:
        policy = EGreedyPolicy(beta)

    if wandb.config.algorithm == 'Q-Learning':
        algo = QLearning(env, gamma, policy) # Could be sarsa too
    else:
        algo = SARSA(env, gamma, policy)
    
    episodes = 10000


    num_expts = 5
    reward_avgs, steps_avgs = [], []

    for i in range(num_expts):
        print("Experiment: %d"%(i+1))
        Q = np.zeros((env.grid.shape[0], env.grid.shape[1], len(env.action_space)))
        rg = np.random.RandomState(i)
        Q, rewards, steps = algo.train(alpha, episodes)
        reward_avgs.append(rewards)
        steps_avgs.append(steps)

    reward_avgs = np.array(reward_avgs)
    steps_avgs = np.array(steps_avgs)

    avg_steps_q = np.mean(steps_avgs, axis=0)
    avg_rewards_q = np.mean(reward_avgs, axis=0)

    

    # wandb.log({"mean_steps_to_completion": sum(steps_to_completion)/len(steps_to_completion),
    #            "mean_episode_reward": sum(episode_rewards)/len(episode_rewards)})
               

sweep_agent = wandb.agent(sweep_id, function=train)


sweep_agent.execute()
