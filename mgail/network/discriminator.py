from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import mgail.common as common


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, size: List[int], do_keep_prob: float):
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
            nn.ReLU(inplace=True),
            nn.Linear(self.arch_params['n_hidden_0'], self.arch_params['n_hidden_1']),
            nn.ReLU(inplace=True)
        )
        self.fc_out = nn.Linear(self.arch_params['n_hidden_1'], out_dim)

        # Initialize weights
        self.model.apply(common.init_weights)
        self.fc_out.apply(common.init_weights)

    def forward(self, state: torch.Tensor, action: torch.Tensor, do_keep_prob: float = None) -> torch.Tensor:
        bs = state.size(0)

        concat = torch.cat([state[:,-1].view(bs, -1), action.view(bs, -1)], axis=1)
        x = self.model(concat)
        
        do_keep_prob = self.arch_params['do_keep_prob'] if do_keep_prob == None else do_keep_prob
        x = F.dropout(x, p=1.0-do_keep_prob, training=self.training)
        
        x = self.fc_out(x)
        return x