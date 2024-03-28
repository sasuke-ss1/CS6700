import torch.nn as nn
import torch
from torch import Tensor

class DDQN(nn.Module):
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

        adv = torch.max(advantage, dim=1, keepdim=True) if self.use_max else torch.mean(advantage, dim=1, keepdim=True)
        #adv = torch.mean(advantage, dim=1, keepdim=True)
        Q_value = value + advantage - adv

        return Q_value

    def select_action(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            Q_value = self.forward(state)
            action = Q_value.argmax(dim=1)

        return action.item()
    
    
# ANK
class REINFORCE(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    # If you can write this then write else lets discuss
    def get_action(self):
        pass 