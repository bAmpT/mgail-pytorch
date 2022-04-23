from collections import OrderedDict
import torch
import torch.nn as nn
import mgail.common as common


class ForwardModel(nn.Module): 
    def __init__(self, state_size: int, action_size: int, encoding_size: int) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.encoding_size = encoding_size

        self.fc_state_encoder = nn.Sequential(
            nn.Linear(state_size, encoding_size),
            nn.ReLU()
        )
        self.rnn = nn.GRUCell(encoding_size, encoding_size)
        self.fc_state_decoder = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.Sigmoid(),
        )

        self.fc_action_encoder = nn.Sequential(
            nn.Linear(action_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.Sigmoid()
        )

        self.fc_dynamics = nn.Sequential(
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, state_size),
        )

        # Initialize weights
        self.fc_state_encoder.apply(common.init_weights)
        self.rnn.apply(common.init_weights)
        self.fc_state_decoder.apply(common.init_weights)
        self.fc_action_encoder.apply(common.init_weights)
        self.fc_dynamics.apply(common.init_weights)

    def forward(self, state, action, gru_state):
        # State embedding
        input = self.fc_state_encoder(state)
        gru_state = self.rnn(input, gru_state)
        encoded_state = self.fc_state_decoder(gru_state)
        
        # Action embedding
        encoded_action = self.fc_action_encoder(action)

        # Joint embedding
        joint_embedding = encoded_state * encoded_action

        # Next state prediction
        next_state = self.fc_dynamics(joint_embedding)

        return next_state, gru_state
