import pickle
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import mgail.common as common
from mgail.env import Environment
from mgail.algo.mgail import MGAIL
import mgail.buffer as buffer
from mgail.buffer import ER

# Load buffer from different repo
import sys
sys.modules['er'] = buffer
sys.modules['ER'] = buffer


class Trainer(object):
    def __init__(self, environment: Environment):
        self.env = environment
        self.algorithm = MGAIL(environment=self.env, gamma=self.env.gamma, do_keep_prob=self.env.do_keep_prob, temp=self.env.temp)
        if self.env.trained_model:
            self.algorithm = torch.load(self.env.trained_model)
        self.run_dir = self.env.run_dir
        self.writer = SummaryWriter(log_dir=f"./runs/{self.env.name}"+time.strftime("-%Y-%m-%d-%H-%M"))
        self.log_name = "./logs/" + self.env.name + time.strftime("-%Y-%m-%d-%H-%M") + ".log" 
        self.loss = 999. * np.ones(3)
        self.reward_mean = 0
        self.reward_std = 0
        self.run_avg = 0.001
        self.discriminator_policy_switch = 0
        self.policy_loop_time = 0
        self.disc_acc = 0
        self.er_count = 0
        self.itr = 0
        self.best_reward = 0
        self.mode = 'Prep'
        self.forward_opt = torch.optim.Adam(self.algorithm.forward_model.parameters(), lr=self.env.fm_lr)
        self.disc_opt = torch.optim.Adam(self.algorithm.discriminator.parameters(), lr=self.env.d_lr, weight_decay=self.env.weight_decay)
        self.policy_opt = torch.optim.Adam(self.algorithm.policy.parameters(), lr=self.env.p_lr, weight_decay=self.env.weight_decay)
        np.set_printoptions(precision=2)
        np.set_printoptions(linewidth=220)

    def update_stats(self, module, attr, value) -> None:
        v = {'forward_model': 0, 'discriminator': 1, 'policy': 2}
        module_ind = v[module]
        if attr == 'loss':
            self.loss[module_ind] = self.run_avg * self.loss[module_ind] + (1 - self.run_avg) * np.asarray(value)
        elif attr == 'accuracy':
            self.disc_acc = self.run_avg * self.disc_acc + (1 - self.run_avg) * np.asarray(value)

    def train_forward_model(self) -> None:
        alg = self.algorithm
        states_, actions, _, states = alg.er_agent.sample()[:4]
        states_, actions, states = torch.as_tensor(states_), torch.as_tensor(actions), torch.as_tensor(states)
        
        states_ = common.normalize(states_, alg.er_expert.states_mean, alg.er_expert.states_std).float()
        actions = common.normalize(actions, alg.er_expert.actions_mean, alg.er_expert.actions_std).float()
        states = common.normalize(states, alg.er_expert.states_mean, alg.er_expert.states_std)
        
        initial_gru_state = torch.ones((states.shape[0], alg.forward_model.encoding_size))
        forward_model_prediction, _ = alg.forward_model(states_, actions, initial_gru_state)
        
        forward_model_loss = torch.mean(torch.square(states-forward_model_prediction)) #F.mse_loss(forward_model_prediction, states) 
        
        self.forward_opt.zero_grad()
        forward_model_loss.backward()
        self.forward_opt.step()

        self.update_stats('forward_model', 'loss', forward_model_loss.item())

    def train_discriminator(self) -> None:
        alg = self.algorithm
        # get states and actions
        state_a_, action_a = alg.er_agent.sample()[:2]
        state_e_, action_e = alg.er_expert.sample()[:2]
        states = torch.as_tensor(np.concatenate([state_a_, state_e_]))
        actions = torch.as_tensor(np.concatenate([action_a, action_e]))
        # labels (policy/expert) : 0/1, and in 1-hot form: policy-[1,0], expert-[0,1]
        labels_a = np.zeros(shape=(state_a_.shape[0],), dtype=np.float32)
        labels_e = np.ones(shape=(state_e_.shape[0],), dtype=np.float32)
        label = torch.as_tensor(np.expand_dims(np.concatenate([labels_a, labels_e]), axis=1))

        labels = torch.cat([1 - label, label], 1)

        states = common.normalize(states, alg.er_expert.states_mean, alg.er_expert.states_std).float()
        actions = common.normalize(actions, alg.er_expert.actions_mean, alg.er_expert.actions_std).float()
        d = alg.discriminator(states, actions)

        # 2.1 0-1 accuracy
        correct_predictions = torch.argmax(d, axis=1) == torch.argmax(labels, axis=1)
        discriminator_acc = torch.mean(correct_predictions.float())
        
        # 2.2 prediction
        d_cross_entropy = (-labels * F.log_softmax(d, dim=-1)).sum(dim=-1) # -torch.sum(F.log_softmax(d, dim=1) * labels, dim=1) # F.cross_entropy(d, labels)

        # cost sensitive weighting (weight true=expert, predict=agent mistakes)
        d_loss_weighted = self.env.cost_sensitive_weight * (label.squeeze() == 1.).float() * d_cross_entropy +\
                                                           (label.squeeze() == 0.).float() * d_cross_entropy
        discriminator_loss = torch.mean(d_loss_weighted)
        self.disc_opt.zero_grad()
        discriminator_loss.backward()
        self.disc_opt.step()

        self.update_stats('discriminator', 'loss', discriminator_loss.item())
        self.update_stats('discriminator', 'accuracy', discriminator_acc.item())

    def train_policy(self) -> None:
        alg = self.algorithm

        # reset the policy gradient
        self.policy_opt.zero_grad()

        # Adversarial Learning
        if self.env.get_status():
            state = self.env.reset()
        else:
            state = self.env.get_state()

        # Accumulate the (noisy) adversarial gradient
        policy_loss = torch.zeros((1))
        for i in range(self.env.policy_accum_steps):
            # accumulate AL gradient
            states = torch.as_tensor(state).unsqueeze(0).float()
            policy_loss += alg.train(states) / self.env.policy_accum_steps
            # clip gradients
            policy_loss.register_hook(lambda grad: torch.clamp(grad, -2, 2))
            
            self.update_stats('policy', 'loss', policy_loss.item())
                    
        # apply AL gradient
        policy_loss.backward() 
        self.policy_opt.step()

    def collect_experience(self, record: bool = 1, vis: bool=0, n_steps: int = None, \
        noise_flag: bool = True, start_at_zero: bool = True) -> float:

        alg = self.algorithm

        # environment initialization point (random or from start)
        # TODO: option to start at zero 
        observation = self.env.reset()

        do_keep_prob = self.env.do_keep_prob
        t = 0
        R = 0
        done = 0
        if n_steps is None:
            n_steps = self.env.n_steps_test

        while not done:
            if vis:
                self.env.render()

            if not noise_flag:
                do_keep_prob = 1.

            states = torch.as_tensor(np.reshape(observation, [1, -1])).float()
            a = alg.action_test(states, noise=float(noise_flag), do_keep_prob=do_keep_prob).detach().cpu().numpy().astype(np.float32)
            
            observation, reward, done, info = self.env.step(a, mode='python')

            done = done or t > n_steps
            t += 1
            R += reward

            if record:
                if self.env.continuous_actions:
                    action = [a]
                else:
                    action = np.zeros((1, self.env.action_size))
                    action[0, a[0]] = 1
                alg.er_agent.add(
                    actions=action, 
                    rewards=[reward], 
                    next_states=[observation], 
                    terminals=[done], 
                )

        return R

    def train_step(self) -> None:
        # phase_1 - Adversarial training
        # forward_model: learning from agent data
        # discriminator: learning in an interleaved mode with policy
        # policy: learning in adversarial mode

        # Fill Experience Buffer
        if self.itr == 0:
            if self.env.agent_data != None:
                with open(self.env.agent_data, 'rb') as f:
                    self.algorithm.er_agent = pickle.load(f)
            else:
                while self.algorithm.er_agent.current == self.algorithm.er_agent.count:
                    self.collect_experience()
                    buf = 'Collecting examples...%d/%d' % \
                        (self.algorithm.er_agent.current, self.algorithm.er_agent.states.shape[0])
                    sys.stdout.write('\r' + buf)

        # Adversarial Learning
        else:
            self.train_forward_model()

            self.mode = 'Prep'
            if self.itr < self.env.prep_time:
                self.train_discriminator()
            else:
                self.mode = 'AL'

                if self.discriminator_policy_switch:
                    self.train_discriminator()
                else:
                    self.train_policy()

                if self.itr % self.env.collect_experience_interval == 0:
                    self.collect_experience(start_at_zero=False, n_steps=self.env.n_steps_train)

                # switch discriminator-policy
                if self.itr % self.env.discr_policy_itrvl == 0:
                    self.discriminator_policy_switch = not self.discriminator_policy_switch

        # print progress
        if self.itr % 100 == 0:
            self.print_info_line('slim')

    def print_info_line(self, mode: str) -> None:
        if mode == 'full':
            self.writer.add_scalar(f"train/reward_mean", self.reward_mean, self.itr)
            self.writer.add_scalar(f"train/reward_std", self.reward_std, self.itr)
            self.writer.add_scalar(f"train/loss_forward_model", self.loss[0], self.itr)
            self.writer.add_scalar(f"train/loss_discriminator", self.loss[1], self.itr)
            self.writer.add_scalar(f"train/loss_policy", self.loss[2], self.itr)
            self.writer.add_scalar(f"train/accuracy_discriminator", self.disc_acc, self.itr)

            buf = '%s Training(%s): iter %d, loss: %s R: %.1f, R_std: %.2f\n' % \
                  (time.strftime("%H:%M:%S"), self.mode, self.itr, self.loss, self.reward_mean, self.reward_std)

            with open(self.log_name, 'a') as f:
                f.write(buf)
        else:
            buf = "processing iter: %d, loss(forward_model,discriminator,policy): %s" % (self.itr, self.loss)
        sys.stdout.write('\r' + buf)

    def save_model(self, dir_name: str = None):
        import os
        if dir_name is None:
            dir_name = self.run_dir + '/snapshots/'
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        fname = dir_name + self.env.name + time.strftime("-%Y-%m-%d-%H-%M-") + ('%0.6d.pth' % self.itr)
        torch.save(self.algorithm, fname)
