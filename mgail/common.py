import pickle
import torch
import torch.nn as nn
import numpy as np

import mgail.buffer as buffer
from mgail.buffer import ER

# Load buffer from different repo
import sys
sys.modules['er'] = buffer
sys.modules['ER'] = buffer

# Experience buffer 

def load_er(fname: str, batch_size: int, history_length: int, traj_length: int) -> ER:
    f = open(fname, 'rb')
    try: 
        er = pickle.load(f)
    except: 
        f.seek(0)
        er = pickle.load(f, encoding="latin1")
    er.batch_size = batch_size
    er = set_er_stats(er, history_length, traj_length)
    return er

def set_er_stats(er: ER, history_length: int, traj_length: int) -> ER:
    state_dim = er.states.shape[-1]
    action_dim = er.actions.shape[-1]
    er.prestates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.poststates = np.empty((er.batch_size, history_length, state_dim), dtype=np.float32)
    er.traj_states = np.empty((er.batch_size, traj_length, state_dim), dtype=np.float32)
    er.traj_actions = np.empty((er.batch_size, traj_length-1, action_dim), dtype=np.float32)
    er.states_min = np.min(er.states[:er.count], axis=0)
    er.states_max = np.max(er.states[:er.count], axis=0)
    er.actions_min = np.min(er.actions[:er.count], axis=0)
    er.actions_max = np.max(er.actions[:er.count], axis=0)
    er.states_mean = np.mean(er.states[:er.count], axis=0)
    er.actions_mean = np.mean(er.actions[:er.count], axis=0)
    er.states_std = np.std(er.states[:er.count], axis=0)
    er.states_std[er.states_std == 0] = 1
    er.actions_std = np.std(er.actions[:er.count], axis=0)
    return er


def re_parametrization(state_e: torch.Tensor, state_a: torch.Tensor) -> torch.Tensor:
    nu = state_e - state_a
    nu = nu.detach()
    return state_a + nu, nu

def normalize(x, mean, std):
    return (x - mean)/std

def denormalize(x, mean, std):
    return x * std + mean


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape,minval=0,maxval=1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(torch.shape(logits))
    return torch.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=True):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = torch.shape(logits)[-1]
        #y_hard = torch.cast(torch.one_hot(torch.argmax(y,1),k), y.dtype)
        y_hard = y == torch.max(y, 1, keep_dims=True)
        y = (y_hard - y).detach() + y
    return y

# other utils

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.15)
        #m.bias.data.fill_(0.01)

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)

def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False

def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)

#----

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

    return gaussian_log_probs - torch.log(
        1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize(means, log_stds):
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, actions)

def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)
