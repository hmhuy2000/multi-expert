import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

from .utils import build_mlp

class StateFunction(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_shape[0]+add_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states):
        return self.net(states)
    
class StateActionFunction(nn.Module):

    def __init__(self, state_shape,action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_shape[0]+action_shape[0]+add_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states,actions):
        return self.net(torch.cat((states,actions),dim=-1))
 
class FailureFunction(nn.Module):
    def __init__(self, state_shape,action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0,num_class=0):
        super().__init__()
        self.net = build_mlp(
            input_dim=state_shape[0]+action_shape[0]+add_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout=0.3
        )
        self.classifier = build_mlp(
            input_dim=1,
            output_dim=num_class,
            hidden_units=[],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            dropout=0.3
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    def forward(self, states,actions):
        return self.net(torch.cat((states,actions),dim=-1))
    
    def get_falure_state_action_score(self, states, actions):
        return F.sigmoid(self.forward(states,actions))*0.01
    
    def get_falure_trajectory_score(self, states,actions):
        batch_size = states.shape[0]
        scores = self.get_falure_state_action_score(states, actions)
        scores = torch.reshape(scores, (batch_size,-1))
        scores = torch.sum(scores,dim=-1,keepdim=True)
        return scores

    def classify_trajectory(self,states,actions):
        scores = self.get_falure_trajectory_score(states,actions)
        predicts = self.classifier(scores)
        return predicts,scores
    
    
class TwinnedStateActionFunction(nn.Module):

    def __init__(self, state_shape,action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(),output_activation=None,add_dim=0):
        super().__init__()
        self.net1 = build_mlp(
            input_dim=state_shape[0]+action_shape[0]+add_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.net2 = build_mlp(
            input_dim=state_shape[0]+action_shape[0]+add_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states, actions):
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states, actions):
        return self.net1(torch.cat([states, actions], dim=-1))