import sys
sys.path.append('..')
sys.path.append('./')
#------------------------------------------#
def main():
    import safety_gymnasium
    from Sources.utils import VectorizedWrapper

    sample_env = safety_gymnasium.make('SafetyPointPush1-v0')
    env = [safety_gymnasium.make(id='SafetyPointPush1-v0') for _ in range(10)]
    env = VectorizedWrapper(env)
    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    device = 'cuda'
    sample_env.close()

    #------------------------------------------#
    from Sources.networks.policy import StateIndependentPolicy
    from Sources.buffers.rollout_buffers import Trajectory_Buffer
    from tqdm import trange
    import torch
    from torch import nn
    import numpy as np

    def exploit(actor,state):
        state = torch.tensor(state, dtype=torch.float, device=device)
        with torch.no_grad():
            action = actor(state)
        return action.cpu().numpy()

    expert_actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256,256,256],
            hidden_activation=nn.Tanh()
        ).to(device)
    expert_actor.load_state_dict(torch.load(
        './weights/PPO/(1.621)-(1.00)-(16.21)-(46.90)/actor.pth'
    ))
    expert_actor.eval()
    buffer_size = 5
    rollout_traj_buffer = Trajectory_Buffer(
        buffer_size=buffer_size,
        traj_len=1000, 
        state_shape=state_shape, 
        action_shape=action_shape, 
        device=device,
    )

    while(rollout_traj_buffer._n<buffer_size):
        print(f'{rollout_traj_buffer._n}/{buffer_size}',end='\r')
        state,_ = env.reset()
        episode_return = np.array([0.0 for _ in range(10)])
        episode_cost = np.array([0.0 for _ in range(10)])
        tmp_buffer = [[] for _ in range(10)]

        for iter in range(1000):
            action = exploit(expert_actor,state)
            next_state, reward, cost, done, _, _ = env.step(action)
            for idx in range(10):
                tmp_buffer[idx].append((state[idx], action[idx], next_state[idx]))
            episode_return += reward
            episode_cost += cost
            state = next_state
        
        for idx in range(10):
            if (rollout_traj_buffer._n==buffer_size):
                break
            if (episode_return[idx]<10):
                continue
            arr_states = []
            arr_actions = []
            arr_next_states = []
            for (tmp_state,tmp_action,tmp_next_state) in tmp_buffer[idx]:
                arr_states.append(tmp_state)
                arr_actions.append(tmp_action)
                arr_next_states.append(tmp_next_state)
            arr_states = np.array(arr_states)
            arr_actions = np.array(arr_actions)
            arr_next_states = np.array(arr_next_states)
            rollout_traj_buffer.append(arr_states,arr_actions,episode_return[idx],
                    episode_cost[idx], arr_next_states)
            
    print(rollout_traj_buffer.total_rewards.mean(),rollout_traj_buffer.total_costs.mean())
    rollout_traj_buffer.save(f'./weights/buffers/{buffer_size}.pt')
    env.close()

if __name__ == '__main__':
    main()