import os
import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n             = 0
        self._p             = 0
        self.buffer_size    = buffer_size
        self.total_size     = buffer_size

        self.states         = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards        = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.total_rewards  = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.costs          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones          = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis        = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states    = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward,total_reward,cost, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p]           = float(reward)
        self.total_rewards[self._p]     = float(total_reward)
        self.costs[self._p]             = float(cost)
        self.dones[self._p]             = float(done)
        self.log_pis[self._p]           = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        start = 0
        idxes = slice(start, self._n)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.costs[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.total_rewards[idxes],
            self.costs[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
    
class Trajectory_Buffer:
    def __init__(self, buffer_size,traj_len, state_shape, action_shape, device):
        self._n             = 0
        self._p             = 0
        self.buffer_size    = buffer_size
        self.total_size     = buffer_size

        self.device = device
        self.states         = torch.empty(
            (self.total_size, traj_len, *state_shape), dtype=torch.float, device=device)
        self.actions        = torch.empty(
            (self.total_size, traj_len, *action_shape), dtype=torch.float, device=device)
        self.total_rewards  = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.total_costs    = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states    = torch.empty(
            (self.total_size, traj_len, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action,total_reward,total_cost, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.total_rewards[self._p]     = float(total_reward)
        self.total_costs[self._p]       = float(total_cost)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.total_rewards[idxes],
            self.total_costs[idxes],
            self.next_states[idxes]
        )
    
    def load(self,path):
        tmp = torch.load(path)
        self._n = tmp['states'].size(0)
        assert self.total_size==self._n
        self.states             = tmp['states'].clone().to(self.device)
        self.actions            = tmp['actions'].clone().to(self.device)
        self.total_rewards      = tmp['total_rewards'].clone().to(self.device)
        self.total_costs        = tmp['total_costs'].clone().to(self.device)
        self.next_states        = tmp['next_states'].clone().to(self.device)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({
            'states': self.states.clone().cpu(),
            'actions': self.actions.clone().cpu(),
            'total_rewards': self.total_rewards.clone().cpu(),
            'total_costs': self.total_costs.clone().cpu(),
            'next_states': self.next_states.clone().cpu(),
        }, path)