import numpy as np
from multiprocessing import Process, Pipe
import multiprocessing as mp

def worker(remote, parent_remote, env):
  '''
  Worker function which interacts with the environment over the remove connection

  Args:
    remote (multiprocessing.Connection): Worker remote connection
    parent_remote (multiprocessing.Connection): MultiRunner remote connection
    env_fn (function): Creates a environment
    planner_fn (function): Creates the planner for the environment
  '''
  parent_remote.close()

  try:
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        res = env.step(data)
        remote.send(res)
      elif cmd == 'reset':
        obs = env.reset()
        remote.send(obs)
      elif cmd =='render':
        env.render()
      elif cmd == 'close':
        remote.close()
        break
      else:
        raise NotImplementedError
  except KeyboardInterrupt:
    print('MultiRunner worker: caught keyboard interrupt')

class VectorizedWrapper(object):
    def __init__(self, envs):
        self.waiting = False
        self.closed = False
        num_envs = len(envs)
        ctx = mp.get_context('spawn')

        self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.processes = [Process(target=worker, args=(worker_remote, remote, env))
                    for (worker_remote, remote, env) in zip(self.worker_remotes, self.remotes, envs)]
        self.num_processes = len(self.processes)

        for process in self.processes:
            process.daemon = True
            process.start()
        for remote in self.worker_remotes:
            remote.close()

    def reset(self):
        '''
        Reset each environment.

        Returns:
        numpy.array: Observations
        '''
        for remote in self.remotes:
            remote.send(('reset', None))
        self.waiting = True

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        res = tuple(zip(*results))
        obs, info = res
        obs = np.stack(obs)
        return obs,info

    def reset_envs(self, env_nums):
        '''
        Resets the specified environments.

        Args:
        env_nums (list[int]): The environments to be reset

        Returns:
        numpy.array: Observations
        '''
        for env_num in env_nums:
            self.remotes[env_num].send(('reset', None))
        self.waiting = True

        results = [self.remotes[env_num].recv() for env_num in env_nums]
        self.waiting = False
        res = tuple(zip(*results))
        obs, info = res
        obs = np.stack(obs)
        return obs,info

    def step(self, actions):
        '''
        Step the environments synchronously.

        Args:
        actions (numpy.array): Actions to take in each environment
        auto_reset (bool): Reset environments automatically after an episode ends
        '''
        self.stepAsync(actions)
        return self.stepWait()

    def stepAsync(self, actions):
        '''
        Step each environment in a async fashion.

        Args:
        actions (numpy.array): Actions to take in each environment
        auto_reset (bool): Reset environments automatically after an episode ends
        '''

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def stepWait(self):
        '''
        Wait until each environment has completed its next step.
        '''
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        res = tuple(zip(*results))

        obs, reward, c, done, _, _ = res
        obs = np.stack(obs)
        reward = np.stack(reward)
        done = np.stack(done)
        c = np.stack(c)
        return obs, reward, c, done, _, _

    def close(self):
      for remote in self.remotes:
        remote.send(('close', None))