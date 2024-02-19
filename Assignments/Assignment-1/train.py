from algo import *
from grid_world import GridWorld
import numpy as np
from policy import SoftmaxPolicy, EGreedyPolicy
from algo import QLearning,SARSA
import matplotlib.pyplot as plt

# Defining parameters

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


epsilon = 0.4
alpha = 0.4
tau = 0.01
gamma = 0.8

#print(env.P[36, 4])
#exit()
# Defining env and algorithms

policy = EGreedyPolicy(epsilon)
q_learning = QLearning(env, gamma, policy)


# Training
episodes = 1000
Q, episode_rewards, steps_to_completion = q_learning.train(alpha, episodes)
q_learning.plot(episode_rewards, steps_to_completion)
plt.figure()
plt.plot(steps_to_completion)
plt.xlabel('Episode')
plt.ylabel('Number of steps to Goal')
# plt.legend()
plt.show()

plt.figure()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
# plt.legend()
plt.show()
