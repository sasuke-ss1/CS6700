import torch.nn as nn
import torch
from torch import Tensor

class DDQN(nn.Module):
    '''
    The architecture for the Q network used in ddqn
    '''
    def __init__(self, state_size: int, action_size: int, hidden_size: list[int], activation="ReLU", use_max=False) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size[0])
        self.fc_value = nn.Linear(hidden_size[0], hidden_size[1]) #Hardcoded
        self.fc_adv = nn.Linear(hidden_size[0], hidden_size[1]) #Hardcoded
        self.act = getattr(nn, activation)()

        self.val = nn.Linear(hidden_size[1], 1)
        self.adv = nn.Linear(hidden_size[1], action_size)
        self.use_max = use_max

    def forward(self, state: Tensor) -> Tensor:
        y = self.act(self.fc1(state))
        value = self.act(self.fc_value(y))
        advantage = self.act(self.fc_adv(y))        

        value = self.val(value)
        advantage = self.adv(advantage)

        adv = torch.max(advantage, dim=1, keepdim=True).values if self.use_max else torch.mean(advantage, dim=1, keepdim=True)
        
        Q_value = value + advantage - adv

        return Q_value

    def select_action(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            Q_value = self.forward(state)
            action = Q_value.argmax(dim=1)

        return action.item()
    
    
class Policy(nn.Module):
    '''
    Policy network for the REINFORCE algorithm
    '''
    def __init__(self, state_size: int, action_size: int, hidden_size: int, activation='ReLU') -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
        
        self.act = getattr(nn, activation)()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        return self.softmax(self.out(x))
    
class Value(nn.Module):
    '''
    Value network for the REINFORCE algorithm
    '''
    def __init__(self, state_size: int, hidden_size: int, activation='ReLU') -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
    
        self.act = getattr(nn, activation)() 

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
       
        return self.out(x)