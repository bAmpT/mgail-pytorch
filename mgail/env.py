import torch
import numpy as np
import gym
from mgail.common import FrameStack


class Environment(object):
    def __init__(self, run_dir: str, env_name: str):
        self.name = env_name
        self._train_params()
        self.gym = gym.make(self.name)
        self.gym = FrameStack(self.gym, self.history_length)
        self.random_initialization = True
        self._connect()
        self.run_dir = run_dir

    def _step(self, action: np.ndarray):
        #action = [np.squeeze(action)]
        self.t += 1
        result = self.gym.step(action)
        self.state, self.reward, self.done, self.info = result[:4]
        return np.float32(self.state), np.float32(self.reward), self.done

    def step(self, action: np.ndarray, mode: str):
        if mode == 'tensorflow':
            state, reward, done = self._step(action)
            state = torch.as_tensor(state).view(1, self.history_length, self.state_size).squeeze()
        else:
            state, reward, done = self._step(action)

        return state, reward, done, 0.

    def reset(self, qpos=None, qvel=None):
        self.t = 0
        self.state = self.gym.reset()
        if self.random_initialization and qpos is not None and qvel is not None:
            self.gym.env.set_state(qpos, qvel)
        return self.state

    def get_status(self):
        return self.done

    def get_state(self):
        return self.state

    def render(self):
        self.gym.render()

    def _connect(self):
        self.state_size = self.gym.observation_space.shape[0]
        if self.history_length > 1:
            self.state_size = self.gym.observation_space.shape[-1]    
        self.action_size = self.gym.action_space.shape[0]
        self.action_space = np.asarray([None]*self.action_size)

    def _train_params(self):
        self.trained_model = None
        self.train_mode = True
        self.agent_data = 'agent_trajectories/hopper_er_agent.bin'
        self.expert_data = 'expert_trajectories/hopper_er.bin'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = False
        self.save_models = True
        self.config_dir = None
        self.continuous_actions = True

        # Main parameters to play with:
        self.er_agent_size = 50000
        self.prep_time = 1000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        self.discr_policy_itrvl = 100
        self.gamma = 0.99
        self.batch_size = 70
        self.history_length = 10 # past trajectory states
        self.traj_length = 10 # prediction horizon
        self.weight_decay = 1e-7
        self.policy_al_w = 1e-2
        self.policy_tr_w = 1e-4
        self.policy_accum_steps = 7
        self.total_trans_err_allowed = 1000
        self.temp = 1.
        self.cost_sensitive_weight = 0.8
        self.noise_intensity = 6.
        self.do_keep_prob = 0.75

        # Hidden layers size
        self.fm_size = 100
        self.d_size = [200, 100]
        self.p_size = [100, 50]

        # Learning rates
        self.fm_lr = 1e-4
        self.d_lr = 1e-3
        self.p_lr = 1e-4



