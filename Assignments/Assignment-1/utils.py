import matplotlib.pyplot as plt
import numpy as np
from math import floor

# DOWN = 0
# UP = 1
# LEFT = 2
# RIGHT = 3

# def plot_Q(Q, message = "Q plot"):
#     plt.figure(figsize=(7,7))
#     plt.title(message)
#     plt.pcolor(Q.max(-1), edgecolors='k', linewidths=2)
#     plt.colorbar()
#     def x_direct(a):
#         if a in [UP, DOWN]:
#             return 0
#         return 1 if a == RIGHT else -1
#     def y_direct(a):
#         if a in [RIGHT, LEFT]:
#             return 0
#         return 1 if a == UP else -1
#     policy = Q.argmax(-1)
#     policyx = np.vectorize(x_direct)(policy)
#     policyy = np.vectorize(y_direct)(policy)
#     idx = np.indices(policy.shape)
#     plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
#     plt.gca().invert_yaxis()
#     plt.show()

def row_col_to_seq(row_col, num_cols = 10):  #Converts state number to row_column format
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols = 10): #Converts row_column format to state number
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])


def plot_state_visits(state_visits, message="State Visits Heatmap"):

    visits = state_visits[:, :, 0]

    plt.figure(figsize=(8, 6))
    plt.imshow(visits, cmap='PuBu', interpolation='nearest')
    plt.colorbar(label='Number of Visits')
    plt.title(message)
    plt.xticks(np.arange(0, visits.shape[1], 1))
    plt.yticks(np.arange(0, visits.shape[0], 1))

    # Add annotations with visit counts
    # for i in range(visits.shape[0]):
    #     for j in range(visits.shape[1]):
    #         plt.text(j, i, str(visits[i, j]), ha='center', va='center', color='black', fontsize=10)

    plt.show()

      

def plot_Q(Q, message="Q plot"):

    optimal_actions = np.argmax(Q, axis=2)
    grid = np.zeros_like(optimal_actions, dtype='<U2')

    action_symbols = {0: '^', 1: 'v', 2: '<', 3: '>'}

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = action_symbols[optimal_actions[i, j]]

    plt.figure(figsize=(8, 6))
    plt.imshow(np.max(Q, axis=2), cmap='PuBu', interpolation='nearest')
    plt.colorbar(label='Max Q-value')
    plt.title(message)
    plt.xticks(np.arange(0, Q.shape[1], 1))
    plt.yticks(np.arange(0, Q.shape[0], 1))

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j] != '':
                plt.text(j, i, grid[i, j], ha='center', va='center', color='dimgray', fontsize=16)

    plt.show()


