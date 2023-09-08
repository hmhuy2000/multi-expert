import numpy as np
import os
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)
def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False

from .base_algo import Algorithm
from Sources.buffers.rollout_buffers import RolloutBuffer
from Sources.networks.policy import StateIndependentPolicy
from Sources.networks.value import TwinnedStateActionFunction

class SAC_continuous(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma,
                SAC_batch_size, buffer_size, lr_actor, lr_critic, 
                lr_alpha, hidden_units_actor, hidden_units_critic, 
                start_steps, tau,max_episode_length,reward_factor,
                max_grad_norm, primarive=True):
        super().__init__(device, seed, gamma)

        if (primarive):
            self.buffer = RolloutBuffer(
                buffer_size=buffer_size,
                state_shape=state_shape,
                action_shape=action_shape,
                device=device,
            )

            # Actor.
            self.actor = StateIndependentPolicy(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_actor,
                hidden_activation=nn.Tanh()
            ).to(device)

            # Critic.
            self.critic = TwinnedStateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.critic_target = TwinnedStateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)

            soft_update(self.critic_target, self.critic, 1.0)
            disable_gradient(self.critic_target)
            self.alpha = 1.0
            self.num_envs = 1
            self.reward_factor = reward_factor
            self.alpha = torch.tensor(0.0).to(self.device)
            self.alpha.requires_grad = True

            self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
            self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)
            self.optim_alpha = torch.optim.Adam([self.alpha], lr=lr_alpha)

        self.max_grad_norm = max_grad_norm
        self.max_episode_length = max_episode_length
        self.target_entropy = -float(action_shape[0])
        self.SAC_batch_size = SAC_batch_size
        self.start_steps = start_steps
        self.tau = tau
        self.return_reward = []
        self.ep_len = [] 
        self.tmp_buffer = [[] for _ in range(self.num_envs)]
        self.tmp_return_reward = [0 for _ in range(self.num_envs)]

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.SAC_batch_size)
    
    def step(self, env, state, ep_len):
        action, log_pi = self.explore(state)
        next_state, reward, c, done, _, _  = env.step(action)
        reset_arr = []

        for idx in range(self.num_envs):
            ep_len[idx] += 1
            mask = False if ep_len[idx] >= self.max_episode_length else done[idx]
            self.tmp_buffer[idx].append((state[idx], action[idx], reward[idx] * self.reward_factor,
            c[idx], mask, log_pi[idx], next_state[idx]))
            self.tmp_return_reward[idx] += reward[idx]
            if (self.max_episode_length and ep_len[idx]>=self.max_episode_length):
                done[idx] = True
            if (done[idx]):
                reset_arr.append(idx)
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward,self.tmp_return_reward[idx],
                     tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                self.tmp_buffer[idx] = []
                self.ep_len.append(ep_len[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.tmp_return_reward[idx] = 0
        if(len(reset_arr)):
            next_state[reset_arr],_ = env.reset_envs(reset_arr)
            ep_len[reset_arr] = [0 for _ in range(len(reset_arr))]

        return next_state, ep_len
    
    def update(self, log_info):
        self.learning_steps += 1
        states, actions, env_rewards,env_total_rewards, costs, dones, log_pis, next_states =\
              self.buffer.sample(self.SAC_batch_size)
        self.update_critic(
            states, actions, env_rewards, dones, next_states, log_info)
        self.update_actor(states, log_info)
        self.update_target()
        log_info.update({
            'Train/return':np.mean(self.return_reward),
            'Train/epLen':np.mean(self.ep_len),
            'Update/env_rewards':env_rewards.mean().item(),
        })
        self.return_reward = []
        self.ep_len = []

    def update_critic(self, states, actions, rewards, dones, next_states,
                      log_info):
        curr_qs1, curr_qs2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - F.softplus(self.alpha) * log_pis
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        log_info.update({
            'Loss/q1_loss':loss_critic1.item(),
            'Loss/q2_loss':loss_critic2.item(),
            'Critic/value_1_mean':curr_qs1.mean().item(),
            'Critic/value_1_max':curr_qs1.max().item(),
            'Critic/value_1_min':curr_qs1.min().item(),
            'Critic/value_2_mean':curr_qs2.mean().item(),
            'Critic/value_2_max':curr_qs2.max().item(),
            'Critic/value_2_min':curr_qs2.min().item(),
            'Critic/target_value_mean':target_qs.mean().item(),
            'Critic/target_value_max':target_qs.max().item(),
            'Critic/target_value_min':target_qs.min().item(),
        })

    def update_actor(self, states, log_info):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        alpha = F.softplus(self.alpha).detach()
        loss_actor = alpha * log_pis.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        entropy = -log_pis.detach_().mean()
        loss_alpha = -F.softplus(self.alpha) * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        log_info.update({
            'Loss/actor_loss':loss_actor.item(),
            'Loss/entropy_loss':alpha * log_pis.mean().item(),
            'Update/entropy':entropy.item(),
            'Update/alpha':alpha,
            'Update/log_pis':log_pis.mean().item(),
        })

    def update_target(self):
        soft_update(self.critic_target, self.critic, self.tau)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def load_models(self,load_dir):
        if not os.path.exists(load_dir):
            raise
        self.actor.load_state_dict(torch.load(f'{load_dir}/actor.pth'))

    def copyNetworksFrom(self,algo):
        self.actor.load_state_dict(algo.actor.state_dict())

    def save_models(self,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')