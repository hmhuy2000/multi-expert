import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('PPO_training')
training_group.add_argument('--env_name',type=str,default='SafetyPointPush1-v0')
training_group.add_argument('--gamma', type=float, default=0.99)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=0)
training_group.add_argument('--buffer_size',type=int,default=50000)
training_group.add_argument('--hidden_units_actor',type=int,default=256)
training_group.add_argument('--hidden_units_critic',type=int,default=256)
training_group.add_argument('--number_layers',type=int,default=3)
training_group.add_argument('--lr_actor', type=float, default=0.0001)
training_group.add_argument('--lr_critic', type=float, default=0.0001)
training_group.add_argument('--epoch_ppo',type=int,default=80)
training_group.add_argument('--clip_eps', type=float, default=0.2)
training_group.add_argument('--lambd', type=float, default=0.97)
training_group.add_argument('--coef_ent', type=float, default=0.0001)
training_group.add_argument('--max_grad_norm', type=float, default=1.0)
training_group.add_argument('--num_training_step',type=int,default=int(3e7))
training_group.add_argument('--eval_interval',type=int,default=int(3e5))
training_group.add_argument('--num_eval_episodes',type=int,default=100)
training_group.add_argument('--max_episode_length',type=int,default=1000)
training_group.add_argument('--reward_factor',type=float,default=1.0)
training_group.add_argument('--weight_path', type=str, default='./weights/PPO')

training_group.add_argument('--begin_cpu',type=int,default=0)
training_group.add_argument('--end_cpu',type=int,default=96)
training_group.add_argument('--num_envs',type=int,default=25)
training_group.add_argument('--eval_num_envs',type=int,default=25)

#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()
gamma                                   = args.gamma
device                                  = args.device_name
seed                                    = args.seed
buffer_size                             = args.buffer_size

hidden_units_actor                      = []
hidden_units_critic                     = []
for _ in range(args.number_layers):
    hidden_units_actor.append(args.hidden_units_actor)
    hidden_units_critic.append(args.hidden_units_critic)

max_value                               = -np.inf

begin_cpu                               = args.begin_cpu
end_cpu                                 = args.end_cpu
lr_actor                                = args.lr_actor
lr_critic                               = args.lr_critic
epoch_ppo                               = args.epoch_ppo
clip_eps                                = args.clip_eps
lambd                                   = args.lambd
coef_ent                                = args.coef_ent
max_grad_norm                           = args.max_grad_norm
num_training_step                       = args.num_training_step
eval_interval                           = args.eval_interval
num_eval_episodes                       = args.num_eval_episodes
env_name                                = args.env_name
reward_factor                           = args.reward_factor
max_episode_length                      = args.max_episode_length
num_envs                                = args.num_envs
eval_num_envs                           = args.eval_num_envs
weight_path                             = args.weight_path


log_path = f'{weight_path}/log_data'
os.makedirs(weight_path,exist_ok=True)
os.makedirs(log_path,exist_ok=True)

eval_return = open(f'{log_path}/return_{seed}.txt','w')
