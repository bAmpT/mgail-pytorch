import torch
import torch.nn.functional as F

import os
import mgail.common as common
from mgail.buffer import ER
from mgail.env import Environment
from mgail.network.forward_model import ForwardModel
from mgail.network.discriminator import Discriminator
from mgail.network.policy import Policy


class MGAIL(object):
    def __init__(self, environment: Environment, gamma: float, do_keep_prob: float, temp: float) -> None:

        self.env = environment

        # Save parameters
        self.gamma = gamma 
        self.temp = temp 
        self.do_keep_prob = do_keep_prob 

        # Create MGAIL modules
        self.forward_model = ForwardModel(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            encoding_size=self.env.fm_size
        )

        self.discriminator = Discriminator(
            in_dim=self.env.state_size + self.env.action_size,
            out_dim=2,
            size=self.env.d_size,
            do_keep_prob=self.do_keep_prob
        )

        self.policy = Policy(
            in_dim=self.env.state_size,
            out_dim=self.env.action_size,
            size=self.env.p_size,
            do_keep_prob=self.do_keep_prob
        )

        # Create experience buffers
        self.er_agent = ER(
            memory_size=self.env.er_agent_size,
            state_dim=self.env.state_size,
            action_dim=self.env.action_size,
            reward_dim=1,  # stub connection
            batch_size=self.env.batch_size,
            history_length=1
        )

        # Load the experts experience buffer
        self.er_expert = common.load_er(fname=os.path.join(self.env.run_dir, self.env.expert_data),
                                        batch_size=self.env.batch_size,
                                        history_length=1,
                                        traj_length=2)

        self.env.sigma = self.er_expert.actions_std / self.env.noise_intensity

    def action_test(self, states: torch.Tensor, noise: float, do_keep_prob: float) -> torch.Tensor:
        # Normalize the states
        states = common.normalize(states, self.er_expert.states_mean, self.er_expert.states_std).float()

        # Get actions using learned policy
        mu = self.policy(states, do_keep_prob)
        if self.env.continuous_actions:
            a = common.denormalize(mu, torch.as_tensor(self.er_expert.actions_mean), torch.as_tensor(self.er_expert.actions_std))
            eta = torch.normal(mean=0.0, std=torch.as_tensor(self.env.sigma))
            action_test = (a + noise * eta).squeeze()
        else:
            a = common.gumbel_softmax(logits=mu, temperature=self.temp)
            action_test = torch.argmax(a, dimension=1)

        return action_test

    def train(self, states: torch.Tensor) -> torch.Tensor:
        # Normalize the inputs
        states = common.normalize(states, self.er_expert.states_mean, self.er_expert.states_std).float()
        
        # Cost value
        total_cost = torch.zeros((1)).type_as(states)

        state, t, total_trans_err, env_term_sig = states[0, :].unsqueeze(0), 0, 0., False
        while (not env_term_sig) and t < self.env.n_steps_train and \
                total_trans_err < self.env.total_trans_err_allowed:

            mu = self.policy(state)
            
            if self.env.continuous_actions:
                eta = torch.as_tensor(self.env.sigma).float() * torch.normal(mean=torch.zeros_like(mu), std=1.0)
                action = mu + eta
            else:
                action = common.gumbel_softmax_sample(logits=mu, temperature=self.temp)

            # minimize the gap between agent logit (d[:,0]) and expert logit (d[:,1])
            d = self.discriminator(state, action)
            
            cost = self.al_loss(d)

            # add step cost
            total_cost += self.gamma**t * cost

            # get action
            if self.env.continuous_actions:
                a_sim = common.denormalize(action.clone().detach().numpy(), self.er_expert.actions_mean, self.er_expert.actions_std)
            else:
                a_sim = torch.argmax(action.clone().detach().numpy(), dimension=1)

            # get next state
            state_env, _, env_term_sig, = self.env.step(a_sim, mode='tensorflow')[:3]
            state_e = common.normalize(state_env, self.er_expert.states_mean, self.er_expert.states_std).float()
            state_e = state_e.detach() # stop gradient
            
            initial_gru_state = torch.ones((1, self.forward_model.encoding_size))
            state_a, _ = self.forward_model(state, action, initial_gru_state)

            state, nu = common.re_parametrization(state_e=state_e, state_a=state_a)
            total_trans_err += torch.mean(abs(nu))
            t += 1

        return total_cost

    def al_loss(self, d: torch.Tensor) -> torch.Tensor:
        logit_agent, logit_expert = d[:,[0]], d[:,[1]]

        # Cross entropy loss
        labels = torch.cat([torch.zeros_like(logit_agent), torch.ones_like(logit_expert)], 1)
        d_cross_entropy = (-labels * F.log_softmax(d, dim=-1)).sum(dim=-1)

        loss = self.env.policy_al_w * torch.mean(d_cross_entropy)

        return loss 
