from collections import OrderedDict
import random
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

class Encoder(nn.Module):
    def __init__(self, input_size: int, output_size: int, encoding_size: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(input_size, encoding_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoding_size, encoding_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoding_size, output_size)
        )
        
    def forward(self, x):
        return self.state_encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_size: int, output_size: int, encoding_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        
        self.state_decoder = nn.Sequential(
            nn.Linear(input_size, encoding_size), 
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoding_size, encoding_size),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(encoding_size, output_size)
        )

    def forward(self, x):
        return self.state_decoder(x)

class ForwardModelVAE(nn.Module): 
    def __init__(self, state_size: int, action_size: int, encoding_size: int) -> None:
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = encoding_size

        self.dropout = 0.1
        self.nfeature = 128
        self.nz = 32 # num of hidden variables
        self.n_steps = 10 # num of predicted steps
        self.n_inputs = 5 # num of history of states
        self.n_out = state_size

        n_hidden = 128
        #self.encoder = StateEncoder(state_size * self.n_inputs, hidden_size, n_hidden)
        self.encoder = nn.Sequential(
            nn.Linear(state_size * self.n_inputs, n_hidden),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_hidden, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, n_hidden), 
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_hidden, state_size)
        )
        
        self.a_encoder = nn.Sequential(
            nn.Linear(action_size, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(self.nfeature, self.hidden_size)
        )

        self.z_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.nfeature),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nfeature, 2*self.nz)
        )
        self.z_expander = nn.Sequential(
            nn.Linear(self.nz, self.nfeature),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(self.nfeature, self.hidden_size)
        )
        #self.u_network = u_network(self)

        #self.y_encoder = StateEncoder(state_size, hidden_size, n_hidden)
        self.y_encoder = nn.Sequential(
            nn.Linear(state_size, n_hidden),
            nn.Dropout(p=self.dropout, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_hidden, self.hidden_size)
        )

        # Initialize weights
        #self.apply(common.init_weights)
        # self.encoder.apply(common.init_weights)
        # self.decoder.apply(common.init_weights)
        # self.y_encoder.apply(common.init_weights)
        # self.a_encoder.apply(common.init_weights)
        # self.z_network.apply(common.init_weights)
        # self.z_expander.apply(common.init_weights)
    
    def reparameterize(self, mu, logvar, sample):
        if self.training or sample:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample_z(self, bs, method=None, h_state=None):
        z = torch.randn(bs, self.nz).type_as(h_state)
        return z

    def forward_single_step(self, input_states, action, z):
        # encode the inputs (without the action)
        bs = input_states.size(0)

        h_state = self.encoder(input_states.view(bs, -1)) # => [bs, 100]

        z_exp = self.z_expander(z).view(bs, self.hidden_size)
        a_emb = self.a_encoder(action).view(h_state.size())

        # Encodete States, Actions und latent variables z werden zusammen addiert und dann durch den decoder gejagt
        h_joint = h_state + z_exp
        h_joint = h_joint + a_emb
        #h_joint = h_joint + self.u_network(h_joint)

        pred_state = self.decoder(h_joint)
        #pred_state = torch.clamp(pred_state + input_states[:, -1], min=-6, max=6)
        #pred_state = torch.sigmoid(pred_state + input_states[:, -1])
        pred_state = pred_state + input_states[:, -1]

        return pred_state

    def forward(self, inputs, actions, targets, sampling=None, z_dropout=0.0):
        """:params 
        inputs: [bs, n_inputs, state_size]
        targets: [bs, n_steps, state_size]
        :returns
        predictions, kdl_loss
        """
        bs = inputs.size(0)
        
        # Reshape states, targets and action sequences
        input_states = inputs # => [bs, n_inputs, state_size]
        target_states = targets
        actions = actions.view(bs, -1, self.action_size) # => [bs, num_actions, action_size]

        npred = actions.size(1)
        ploss = torch.zeros(1).type_as(inputs)

        pred_states = []
        z_list = []
        z = None

        for t in range(npred):
            # encode the inputs (without the action)
            h_state = self.encoder(input_states.view(bs, -1)) # => [bs, 100]

            # we are training or estimating z distribution
            if sampling is None:
                # Encode the target states as hidden variable
                h_target = self.y_encoder(target_states[:, t]) # => [bs, 100]

                # Use z dropout to decouple action influence
                if random.random() < z_dropout:
                    z = self.sample_z(bs, method=None, h_state=h_state).data
                else:
                    # Predict parameters of latent gaussian distribution 
                    mu_logvar = self.z_network((h_state + h_target).view(bs, -1)) # => [bs, 32]
                    mu_logvar = mu_logvar.view(bs, 2, self.nz) # => [bs, 2, 16]

                    mu = mu_logvar[:, 0]
                    logvar = mu_logvar[:, 1]
                    z = self.reparameterize(mu, logvar, True)
                    logvar = torch.clamp(logvar, max=4)  # this can go to inf when taking exp(), so clamp it
                    
                    # Kullback-Leibler Divergence (KLD) to match normal distribution
                    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    kld /= bs
                    ploss += kld
            else:
                z = self.sample_z(bs, method=None, h_state=h_state)

            z_list.append(z)
            
            # Build joint hidden variable with state, latent z and action embedding 
            z_exp = self.z_expander(z).view(bs, self.hidden_size)
            h_joint = h_state + z_exp
            a_emb = self.a_encoder(actions[:, t]).view(bs, self.hidden_size)
            h_joint = h_joint + a_emb
            #h_joint = h_joint + self.u_network(h_joint)

            # Predict next state with the decoder
            pred_state = self.decoder(h_joint)
            if sampling is not None:
                pred_state.detach()
            # Next state is relative to previous state
            pred_state = pred_state + input_states[:, -1]
            #pred_state = torch.sigmoid(pred_state + input_states[:, -1])
            pred_states.append(pred_state)
            
            # Append predicted states to input for next prediction step
            input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)
            
        pred_states = torch.stack(pred_states, 1) # => [bs, n_pred, 11]
        z_list = torch.stack(z_list, 1) # => [bs, n_pred, 16]
        return [pred_states, z_list], ploss
