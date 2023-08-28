import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np

from Sources.algos.base_algo import Algorithm
from Sources.buffers.rollout_buffers import RolloutBuffer
from Sources.networks.policy import StateIndependentPolicy
from Sources.networks.value import StateFunction

def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = torch.empty_like(rewards)

    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

class PPO_continuous(Algorithm):
    def __init__(self, state_shape, action_shape, device, seed, gamma,
        buffer_size, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent,
        max_grad_norm,reward_factor,max_episode_length,
        num_envs,primarive=True):
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
            self.critic = StateFunction(
                state_shape=state_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)

            self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
            self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.rollout_length = buffer_size
        self.coef_ent       = coef_ent
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.reward_factor = reward_factor
        self.max_episode_length = max_episode_length
        self.return_reward = []
        self.ep_len = [] 
        self.num_envs = num_envs
        self.target_kl = 0.02
        self.tmp_buffer = [[] for _ in range(self.num_envs)]
        self.tmp_return_reward = [0 for _ in range(self.num_envs)]

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

    def is_update(self,step):
        return step % self.rollout_length == 0

    def update(self,log_info):
        self.learning_steps += 1
        states, actions, env_rewards,env_total_rewards, costs, dones, log_pis, next_states = \
            self.buffer.get()
        env_rewards = env_rewards.clamp(min=-3.0,max=3.0)
        rewards = env_rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states,log_info)

        log_info.update({
            'Train/return':np.mean(self.return_reward),
            'Train/epLen':np.mean(self.ep_len),
        })
        self.return_reward = []
        self.ep_len = []

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,log_info):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)
        
        for _ in range(self.epoch_ppo):
            value_loss = self.update_critic(states, targets)
        app_kl = 0.0
        for iter in range(self.epoch_ppo):
            if (app_kl>self.target_kl):
                break
            actor_loss,app_kl,entropy = self.update_actor(states, actions,log_pis, gaes)

        log_info.update({
            'Critic/gae_mean':gaes.mean().item(),
            'Critic/gae_max':gaes.max().item(),
            'Critic/gae_min':gaes.min().item(),
            'Critic/target_value_mean':targets.mean().item(),
            'Critic/target_value_max':targets.max().item(),
            'Critic/target_value_min':targets.min().item(),
            'Train/nUpd':iter,
            'Train/KL':app_kl,
            'Train/entropy':entropy,
            'Loss/value_loss':value_loss,
            'Loss/actor_loss':actor_loss,
        })

    def update_critic(self, states, targets):
        value_means = self.critic(states)
        loss_critic = (value_means - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
        return loss_critic.item()

    def update_actor(self, states, actions, log_pis_old, gaes):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()
        approx_kl = (log_pis_old - log_pis).mean().item()
        ratios = (log_pis - log_pis_old).exp_()

        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()
        total_loss  = loss_actor - self.coef_ent * entropy 
        self.optim_actor.zero_grad()
        (total_loss).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()
        return total_loss.item(),approx_kl,entropy.item()

    def save_models(self,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pth')

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
        self.critic.load_state_dict(torch.load(f'{load_dir}/critic.pth'))

    def copyNetworksFrom(self,algo):
        self.actor.load_state_dict(algo.actor.state_dict())
