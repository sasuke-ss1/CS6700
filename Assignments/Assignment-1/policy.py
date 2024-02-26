import numpy as np
from utils import row_col_to_seq, seq_to_col_row
from scipy.special import softmax

seed = 42
rg = np.random.RandomState(seed)


class Policy(object):
    '''
    Policy parent class
    (Subclassed by each policy)
    '''
    def __init__(self):
        raise NotImplementedError

    def change_params(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        raise NotImplementedError

class SoftmaxPolicy(Policy):
    '''
    Implements the Softmax Policy with a temperature parameter
    '''
    def __init__(self, tau=0):
        self.tau = tau
    
    def change_params(self, params):
        self.tau = params
    
    def __call__(self, Q, state):
        action_probs = softmax(Q[state[0,0], state[0,1]]/self.tau)
        return rg.choice(np.arange(len(action_probs)), p=action_probs)

class EGreedyPolicy(Policy):
    '''
    Implements the E-Greedu Policy
    (Greedy for 1-e fraction of runs)
    '''
    def __init__(self, epsilon=0):
        assert epsilon >= 0 and epsilon <= 1, "Epsilon should be between [0,1]"
        self.epsilon = epsilon
    
    def change_params(self, params):
        self.epsilon = params
    
    def __call__(self, Q,state):
        if rg.rand() < self.epsilon or not Q[state[0,0], state[0,1]].any():
            return rg.randint(0, Q.shape[-1])
        else:
            return np.argmax(Q[state[0,0], state[0,1]])