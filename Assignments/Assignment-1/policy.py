import numpy as np
from utils import row_col_to_seq, seq_to_col_row
from scipy.special import softmax

seed = 42
rg = np.random.RandomState(seed)


class Policy(object):
    def __init__(self):
        raise NotImplementedError

    def change_params(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        raise NotImplementedError

class SoftmaxPolicy(Policy):
    def __init__(self, tau=0):
        self.tau = tau
    
    def change_params(self, params):
        self.tau = params
    
    def __call__(self, Q, state):
        action_probs = softmax(Q[state[0,0], state[0,1]]/self.tau)
        return rg.choice(np.arange(len(action_probs)), p=action_probs)

class egreedy_policy(Policy):
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
    
    def change_params(self, params):
        self.epsilon = params
    
    def __call__(self, Q,state):
        if rg.rand() < self.epsilon or not Q[state[0,0], state[0,1]].any():
            return rg.randint(0, Q.shape[1])
        else:
            state = seq_to_col_row(state)
            return np.argmax(Q[state[0,0], state[0,1]])  