import matplotlib.pyplot as plt
import numpy as np
from math import floor

DOWN = 0
UP = 1
LEFT = 2
RIGHT = 3

def plot_Q(Q, message = "Q plot"):
    plt.figure(figsize=(10,10))
    plt.title(message)
    plt.pcolor(Q.max(-1), edgecolors='k', linewidths=2)
    plt.colorbar()
    def x_direct(a):
        if a in [UP, DOWN]:
            return 0
        return 1 if a == RIGHT else -1
    def y_direct(a):
        if a in [RIGHT, LEFT]:
            return 0
        return 1 if a == UP else -1
    policy = Q.argmax(-1)
    policyx = np.vectorize(x_direct)(policy)
    policyy = np.vectorize(y_direct)(policy)
    idx = np.indices(policy.shape)
    plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
    plt.show()

def row_col_to_seq(row_col, num_cols = 10):  #Converts state number to row_column format
    return row_col[:,0] * num_cols + row_col[:,1]

def seq_to_col_row(seq, num_cols = 10): #Converts row_column format to state number
    r = floor(seq / num_cols)
    c = seq - r * num_cols
    return np.array([[r, c]])



      
# def step(self, state, action):
#     p, r = 0, np.random.random()
#     for next_state in range(self.num_states):
        
#         p += self.P[state, next_state, action]
        
#         if r <= p:
#             break

#     if(self.wind and np.random.random() < 0.4):

#         arr = self.P[next_state, :, 3]
#         next_next = np.where(arr == np.amax(arr))
#         next_next = next_next[0][0]
#         if next_next in self.goal_states_seq:
#             done=True 
#         else:
#             done=False
#         return next_next, self.R[next_next],done
#     else:
#         if next_state in self.goal_states_seq:
#             done=True 
#         else:
#         done=False
#         return next_state, self.R[next_state],done