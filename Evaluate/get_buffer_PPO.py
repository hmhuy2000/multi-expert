import sys
sys.path.append('..')
sys.path.append('./')
#------------------------------------------#
def main():
    import safety_gymnasium
    from Sources.utils import VectorizedWrapper
    num_envs = 25

    sample_env = safety_gymnasium.make('SafetyPointGoal1-v0')
    env = [safety_gymnasium.make(id='SafetyPointGoal1-v0') for _ in range(num_envs)]
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

    def explore(actor,state):
        state = torch.tensor(state, dtype=torch.float, device=device)
        with torch.no_grad():
            (action,log_pi) = actor.sample(state)
        return action.cpu().numpy(),log_pi.cpu().numpy()

    expert_actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256,256,256],
            hidden_activation=nn.Tanh()
        ).to(device)
    # expert_actor.load_state_dict(torch.load(
    #     # './weights/PPO/(1.621)-(1.00)-(16.21)-(46.90)/actor.pth'
    #     # './weights/PPO/(1.135)-(1.00)-(11.35)-(38.87)/actor.pth'
    #     # './weights/PPO/(0.696)-(1.00)-(6.96)-(47.28)/actor.pth'
    #     './weights/PPO/(0.214)-(1.00)-(2.14)-(63.67)/actor.pth'
    # ))
    
    expert_actor.load_state_dict(torch.load(
        './weights/SafetyPointGoal1-v0/PPO/(0.494)-(1.00)-(4.94)-(82.58)/actor.pth'
        # './weights/SafetyPointGoal1-v0/PPO/(1.505)-(1.00)-(15.05)-(68.56)/actor.pth'
        # './weights/SafetyPointGoal1-v0/PPO/(2.099)-(1.00)-(20.99)-(50.44)/actor.pth'
        # './weights/SafetyPointGoal1-v0/PPO/(2.545)-(1.00)-(25.45)-(50.84)/actor.pth'
        # './weights/SafetyPointGoal1-v0/PPO/(2.710)-(1.00)-(27.10)-(52.71)/actor.pth'
    ))
    
    expert_actor.eval()
    buffer_size = 1000
    rollout_traj_buffer = Trajectory_Buffer(
        buffer_size=buffer_size,
        traj_len=1000, 
        state_shape=state_shape, 
        action_shape=action_shape, 
        device=device,
    )

    while(rollout_traj_buffer._n<buffer_size):
        state,_ = env.reset()
        episode_return = np.array([0.0 for _ in range(num_envs)])
        episode_cost = np.array([0.0 for _ in range(num_envs)])
        tmp_buffer = [[] for _ in range(num_envs)]

        for iter in range(1000):
            # action = exploit(expert_actor,state)
            action,log_pi = explore(expert_actor,state)
            next_state, reward, cost, done, _, _ = env.step(action)
            for idx in range(num_envs):
                tmp_buffer[idx].append((state[idx], action[idx], next_state[idx]))
            episode_return += reward
            episode_cost += cost
            state = next_state
        
        for idx in range(num_envs):
            if (rollout_traj_buffer._n==buffer_size):
                break
            # if (episode_return[idx]<10):
            #     continue
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
            
        print(f'{rollout_traj_buffer._n}/{buffer_size},{rollout_traj_buffer.total_rewards.mean():.2f},{rollout_traj_buffer.total_costs.mean():.2f}',end='\r')
            
    print(rollout_traj_buffer.total_rewards.mean(),rollout_traj_buffer.total_costs.mean())
    rollout_traj_buffer.save(f'./buffers/{"SafetyPointGoal1-v0"}/e4/{buffer_size}.pt')
    env.close()

if __name__ == '__main__':
    main()