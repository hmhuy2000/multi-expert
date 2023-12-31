import torch
from torch import nn
from torch.optim import Adam
import numpy as np

from Sources.buffers.rollout_buffers import Trajectory_Buffer
from Sources.algos.PPO import PPO_continuous
from Sources.algos.SAC import SAC_continuous
from Sources.networks.value import StateActionFunction

def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    gaes = torch.empty_like(rewards)

    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)

class MLE_onpolicy(PPO_continuous):
    def __init__(self,expert_buffer, state_shape, action_shape, device, seed, gamma,
        buffer_size, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent,
        max_grad_norm,reward_factor,max_episode_length,
        num_envs,primarive=True):
        super().__init__(state_shape, action_shape, device, seed, gamma,
        buffer_size, hidden_units_actor, hidden_units_critic,
        lr_actor, lr_critic, epoch_ppo, clip_eps, lambd, coef_ent,
        max_grad_norm,reward_factor,max_episode_length,
        num_envs,primarive=True)

        self.expert_buffer = expert_buffer
        if (primarive):
            self.failure_network = StateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.rollout_traj_buffer = Trajectory_Buffer(
                buffer_size=100,
                traj_len=max_episode_length, 
                state_shape=state_shape, 
                action_shape=action_shape, 
                device=device,
            )
            self.optim_failure = Adam(self.failure_network.parameters(), lr=3*lr_critic)
        self.return_cost = []
        self.tmp_return_cost = [0 for _ in range(self.num_envs)]
        self.batch_size = 64
        
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
            self.tmp_return_cost[idx] += c[idx]
            if (self.max_episode_length and ep_len[idx]>=self.max_episode_length):
                done[idx] = True

            if (done[idx]):
                reset_arr.append(idx)
                arr_states = []
                arr_actions = []
                arr_next_states = []
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward,self.tmp_return_reward[idx],
                     tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                    arr_states.append(tmp_state)
                    arr_actions.append(tmp_action)
                    arr_next_states.append(tmp_next_state)
                
                arr_states = np.array(arr_states)
                arr_actions = np.array(arr_actions)
                arr_next_states = np.array(arr_next_states)
                self.rollout_traj_buffer.append(arr_states,arr_actions,self.tmp_return_reward[idx],
                    self.tmp_return_cost[idx], arr_next_states)

                self.tmp_buffer[idx] = []
                self.ep_len.append(ep_len[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.return_cost.append(self.tmp_return_cost[idx])
                self.tmp_return_reward[idx] = 0
                self.tmp_return_cost[idx] = 0
        if (len(reset_arr)):
            next_state[reset_arr],_ = env.reset_envs(reset_arr)
            ep_len[reset_arr] = [0 for _ in range(len(reset_arr))]

        return next_state, ep_len

    def update(self,log_info):
        self.learning_steps += 1
        states, actions, env_rewards,env_total_rewards, costs, dones, log_pis, next_states = \
            self.buffer.get()
        
        for _ in range(1000):
            pi_states,pi_actions,pi_total_rewards,pi_total_costs,pi_next_states = \
                self.rollout_traj_buffer.sample(batch_size=self.batch_size)
            exp_states,exp_actions,exp_total_rewards,exp_total_costs,exp_next_states = \
                self.expert_buffer.sample(batch_size=self.batch_size)
            pi_fail_scores, exp_fail_scores = self.update_failure_network(
                pi_states,exp_states,
                pi_actions,exp_actions,
                pi_total_rewards,exp_total_rewards
                )

        env_rewards = env_rewards.clamp(min=-3.0,max=3.0)
        failure_rewards = -self.failure_network.get_falure_state_action_score(states, actions).detach()
        rewards = failure_rewards
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states,log_info)

        log_info.update({
            'Train/return':np.mean(self.return_reward),
            'Train/cost':np.mean(self.return_cost),
            'Train/epLen':np.mean(self.ep_len),
            'Update/reward':failure_rewards.mean(),
            'Update/pi_fail_scores':pi_fail_scores,
            'Update/exp_fail_scores':exp_fail_scores,
        })
        self.return_reward = []
        self.ep_len = []

    def update_failure_network(self,
                               noise_states,exp_states,
                                noise_actions,exp_actions,
                                noise_total_rewards,exp_total_rewards
                               ):
        noise_fail_scores = self.failure_network.get_falure_trajectory_score(noise_states,noise_actions)
        exp_fail_scores = self.failure_network.get_falure_trajectory_score(exp_states,exp_actions)

        fail_scores = torch.concat((noise_fail_scores,exp_fail_scores),dim=0)
        total_rewards  = torch.concat((noise_total_rewards,exp_total_rewards),dim=0)

        fail_matrix = fail_scores - fail_scores.view(1, -1)
        reward_dist = (total_rewards - total_rewards.view(1, -1))
        factor = 1/((3*reward_dist).clamp(min=1.0).detach())

        loss_failure = (reward_dist.clamp(min=0.0,max=1.0).detach()*(factor*fail_matrix**2 + fail_matrix)).mean()
        loss_expert = (exp_fail_scores**2).mean()
        self.optim_failure.zero_grad()
        (loss_failure+loss_expert).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.failure_network.parameters(), self.max_grad_norm)
        self.optim_failure.step()
        return noise_fail_scores.mean().item(), exp_fail_scores.mean().item()

class MLE_offpolicy(SAC_continuous):
    def __init__(self,expert_buffer, noisy_buffer, state_shape, action_shape, device, seed, gamma,
            SAC_batch_size, buffer_size, lr_actor, lr_critic, 
            lr_alpha, hidden_units_actor, hidden_units_critic, 
            start_steps, tau,max_episode_length,reward_factor,
            max_grad_norm, primarive=True):
        super().__init__(state_shape, action_shape, device, seed, gamma,
                SAC_batch_size, buffer_size, lr_actor, lr_critic, 
                lr_alpha, hidden_units_actor, hidden_units_critic, 
                start_steps, tau,max_episode_length,reward_factor,
                max_grad_norm, primarive=True)
        
        self.expert_buffer = expert_buffer
        self.noisy_buffer = noisy_buffer
        if (primarive):
            self.failure_network = StateActionFunction(
                state_shape=state_shape,
                action_shape=action_shape,
                hidden_units=hidden_units_critic,
                hidden_activation=nn.Tanh()
            ).to(device)
            self.rollout_traj_buffer = Trajectory_Buffer(
                buffer_size=100,
                traj_len=max_episode_length, 
                state_shape=state_shape, 
                action_shape=action_shape, 
                device=device,
            )
            self.optim_failure = Adam(self.failure_network.parameters(), lr=lr_critic)

        self.return_cost = []
        self.tmp_return_cost = [0 for _ in range(self.num_envs)]
        self.batch_size = 32
        self.pi_fail_scores = self.noise_fail_scores = self.exp_fail_scores = self.reward_dist = None

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
                arr_states = []
                arr_actions = []
                arr_next_states = []
                for (tmp_state,tmp_action,tmp_reward,tmp_c,tmp_mask,tmp_log_pi,tmp_next_state) in self.tmp_buffer[idx]:
                    self.buffer.append(tmp_state, tmp_action, tmp_reward,self.tmp_return_reward[idx],
                     tmp_c, tmp_mask, tmp_log_pi, tmp_next_state)
                    arr_states.append(tmp_state)
                    arr_actions.append(tmp_action)
                    arr_next_states.append(tmp_next_state)

                arr_states = np.array(arr_states)
                arr_actions = np.array(arr_actions)
                arr_next_states = np.array(arr_next_states)
                self.rollout_traj_buffer.append(arr_states,arr_actions,self.tmp_return_reward[idx],
                    self.tmp_return_cost[idx], arr_next_states)

                self.tmp_buffer[idx] = []
                self.ep_len.append(ep_len[idx])
                self.return_reward.append(self.tmp_return_reward[idx])
                self.return_cost.append(self.tmp_return_cost[idx])
                self.tmp_return_reward[idx] = 0
                self.tmp_return_cost[idx] = 0
        if(len(reset_arr)):
            next_state[reset_arr],_ = env.reset_envs(reset_arr)
            ep_len[reset_arr] = [0 for _ in range(len(reset_arr))]

        return next_state, ep_len
    
    def update(self, log_info):
        self.learning_steps += 1
        states, actions, env_rewards,env_total_rewards, costs, dones, log_pis, next_states =\
              self.buffer.sample(self.SAC_batch_size)
        pi_states,pi_actions,pi_total_rewards,pi_total_costs,pi_next_states = \
            self.rollout_traj_buffer.sample(batch_size=self.batch_size)
        noise_states,noise_actions,noise_total_rewards,noise_total_costs,noise_next_states = \
            self.noisy_buffer.sample(batch_size=self.batch_size)            
        exp_states,exp_actions,exp_total_rewards,exp_total_costs,exp_next_states = \
            self.expert_buffer.sample(batch_size=self.batch_size)
        self.update_failure_network(
            pi_states,noise_states,exp_states,
            pi_actions,noise_actions,exp_actions,
            pi_total_rewards,noise_total_rewards,exp_total_rewards
            )
        if (self.learning_steps<3000):
            return
        
        failure_rewards = -self.failure_network.get_falure_state_action_score(states, actions).detach()
        rewards = failure_rewards

        self.update_critic(
            states, actions, rewards, dones, next_states, log_info)
        self.update_actor(states, log_info)
        self.update_target()
        log_info.update({
            'Update/pi_fail_scores':self.pi_fail_scores,
            'Update/noise_fail_scores':self.noise_fail_scores,
            'Update/exp_fail_scores':self.exp_fail_scores,
            'Update/reward_dist':self.reward_dist,
            'Update/failure_rewards':failure_rewards.mean().item(),
            'Update/env_rewards':env_rewards.mean().item(),
            'Train/epLen':np.mean(self.ep_len),
            'Train/return':np.mean(self.return_reward),
            })
        self.return_reward = []
        self.ep_len = []

    def update_from_dataset(self, log_info):
        self.learning_steps += 1
        noise_states,noise_actions,noise_total_rewards,noise_total_costs,noise_next_states = \
            self.noisy_buffer.sample(batch_size=self.batch_size)            
        exp_states,exp_actions,exp_total_rewards,exp_total_costs,exp_next_states = \
            self.expert_buffer.sample(batch_size=self.batch_size)
        
        states = torch.concat((noise_states,exp_states),dim=0)
        actions = torch.concat((noise_actions,exp_actions),dim=0)
        next_states = torch.concat((noise_next_states,exp_next_states),dim=0)

        self.update_failure_network(
            noise_states,exp_states,
            noise_actions,exp_actions,
            noise_total_rewards,exp_total_rewards
            )
        if (self.learning_steps<3000):
            return
        
        failure_rewards = -self.failure_network.get_falure_state_action_score(states, actions).detach()
        rewards = failure_rewards

        self.update_critic(
            states, actions, rewards, torch.zeros_like(rewards), next_states, log_info)
        self.update_actor(states, log_info)
        self.update_target()
        log_info.update({
            'Update/noise_fail_scores':self.noise_fail_scores,
            'Update/exp_fail_scores':self.exp_fail_scores,
            'Update/reward_dist':self.reward_dist,
            'Update/failure_rewards':failure_rewards.mean().item(),
            })
        self.return_reward = []
        self.ep_len = []

    def update_failure_network(self,
                               noise_states,exp_states,
                                noise_actions,exp_actions,
                                noise_total_rewards,exp_total_rewards
                               ):
        noise_fail_scores = self.failure_network.get_falure_trajectory_score(noise_states,noise_actions)
        exp_fail_scores = self.failure_network.get_falure_trajectory_score(exp_states,exp_actions)

        fail_scores = torch.concat((noise_fail_scores,exp_fail_scores),dim=0)
        total_rewards  = torch.concat((noise_total_rewards,exp_total_rewards),dim=0)

        fail_matrix = fail_scores - fail_scores.view(1, -1)
        reward_dist = (total_rewards - total_rewards.view(1, -1))
        factor = 1/((2*reward_dist).clamp(min=1.0).detach())

        loss_failure = (reward_dist.clamp(min=0.0,max=1.0).detach()*(factor*fail_matrix**2 + fail_matrix)).mean()
        loss_expert = (exp_fail_scores**2).mean()
        self.optim_failure.zero_grad()
        (loss_failure+loss_expert).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.failure_network.parameters(), self.max_grad_norm)
        self.optim_failure.step()
        self.noise_fail_scores = noise_fail_scores.mean().item()
        self.exp_fail_scores = exp_fail_scores.mean().item()
        self.reward_dist = reward_dist.clamp(min=0.0).mean().item()