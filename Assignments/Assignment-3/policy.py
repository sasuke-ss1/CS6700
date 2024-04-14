import numpy as np
from scipy.special import softmax

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
    def __init__(self, tau, seed):
        self.params = tau
        self.rg = np.random.RandomState(seed)

    def change_params(self, params):
        self.params = params
    
    def __call__(self, Q, state):
        action_probs = softmax(Q[state]/self.params)
        return self.rg.choice(np.arange(len(action_probs)), p=action_probs)

class EGreedyPolicy(Policy):
    '''
    Implements the E-Greedu Policy
    (Greedy for 1-e fraction of runs)
    '''
    def __init__(self, epsilon, seed):
        assert epsilon >= 0 and epsilon <= 1, "Epsilon should be between [0,1]"
        self.params = epsilon
        self.rg = np.random.RandomState(seed)

    def change_params(self, params):
        self.params = params
    
    def __call__(self, Q,state):
        if self.rg.rand() < self.params or not Q[state].any():
            return self.rg.randint(0, Q.shape[-1])
        else:
            return np.argmax(Q[state])