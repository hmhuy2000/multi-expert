import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.PPO import *
#------------------------------------------#
def main():
    import safety_gymnasium
    from Sources.utils import VectorizedWrapper

    sample_env = safety_gymnasium.make(env_name)
    env = [safety_gymnasium.make(id=env_name) for _ in range(num_envs)]
    env = VectorizedWrapper(env)
    if (eval_num_envs):
        test_env = [safety_gymnasium.make(id=env_name) for _ in range(eval_num_envs)]
        test_env = VectorizedWrapper(test_env)
    else:
        test_env = None

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    sample_env.close()

    #------------------------------------------#
    from Sources.algos.PPO import PPO_continuous
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    import wandb

    #------------------------------------------#
    def evaluate(algo, env,max_episode_length):
        global max_value
        mean_return = 0.0
        mean_cost = 0.0
        failed_case = []
        cost_sum = [0 for _ in range(eval_num_envs)]

        for step in range(num_eval_episodes//eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            for iter in range(max_episode_length):
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            mean_return += episode_return 
            mean_cost += episode_cost 

        mean_return = mean_return/num_eval_episodes
        mean_cost = mean_cost/num_eval_episodes

        success_rate = 1.0
        value = (mean_return * success_rate)/10
        if (value>max_value):
            max_value = value
            algo.save_models(f'{weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_value*=0.999

        eval_return.write(f'{mean_return}\n')
        eval_return.flush()

        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_value:.2f}')

    def train(env,test_env,algo,eval_algo):
        t = np.array([0 for _ in range(num_envs)])
        eval_thread = None
        state,_ = env.reset()

        print('start training')
        for step in range(1,num_training_step//num_envs+1):
            if (step%100 == 0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
            state, t = algo.step(env, state, t)
            if algo.is_update(step*num_envs):
                    log_info = {'log_cnt':(step*num_envs)//buffer_size}
                    algo.update(log_info)
                    wandb.log(log_info, step = log_info['log_cnt'])
                    
            if step % (eval_interval//num_envs) == 0:
                algo.save_models(f'{weight_path}/s{seed}-latest')
                if (test_env):
                    if eval_thread is not None:
                        eval_thread.join()
                    eval_algo.copyNetworksFrom(algo)
                    eval_algo.eval()
                    eval_thread = threading.Thread(target=evaluate, 
                    args=(eval_algo,test_env,max_episode_length))
                    eval_thread.start()
        algo.save_models(f'{weight_path}/s{seed}-finish')

    setproctitle.setproctitle(f'{env_name}-PPO-{seed}')
    
    algo = PPO_continuous(
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,buffer_size=buffer_size,
            hidden_units_actor=hidden_units_actor,hidden_units_critic=hidden_units_critic,
            lr_actor=lr_actor,lr_critic=lr_critic, epoch_ppo=epoch_ppo,
            clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,
            max_episode_length=max_episode_length,num_envs=num_envs)
    eval_algo = deepcopy(algo)

    wandb.init(project=f'test-{env_name}', settings=wandb.Settings(_disable_stats=True), \
        group='test', name=f'{seed}', entity='hmhuy')

    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

if __name__ == '__main__':
    main()