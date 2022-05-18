from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import mgail.common as common


class Policy(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, size: List[int], do_keep_prob: float) -> None:
        super().__init__()

        self.arch_params = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'n_hidden_0': size[0],
            'n_hidden_1': size[1],
            'do_keep_prob': do_keep_prob
        }

        self.model = nn.Sequential(
            nn.Linear(self.arch_params['in_dim'], self.arch_params['n_hidden_0']),
            nn.ReLU(),
            nn.Linear(self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(self.arch_params['n_hidden_1'], out_dim)
        
        #self.model.apply(common.init_weights)
        #self.fc_out.apply(common.init_weights)
       
    def forward(self, state: torch.Tensor, do_keep_prob: float = None) -> torch.Tensor:
        bs = state.size(0)

        x = self.model(state[:,-1].view(bs, -1))
        
        do_keep_prob = self.arch_params['do_keep_prob'] if do_keep_prob == None else do_keep_prob
        x = F.dropout(x, p=1.0-do_keep_prob, training=self.training)
        
        x = self.fc_out(x)
        return x
    

class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = common.build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return common.reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return common.evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = common.build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return common.reparameterize(means, log_stds.clamp_(-20, 2))