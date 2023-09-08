import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('SAC_training')
training_group.add_argument('--env_name',type=str,default='SafetyPointButton1-v0')
training_group.add_argument('--gamma', type=float, default=0.99)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=0)
training_group.add_argument('--buffer_size',type=int,default=int(1e6))
training_group.add_argument('--hidden_units_actor',type=int,default=256)
training_group.add_argument('--hidden_units_critic',type=int,default=256)
training_group.add_argument('--number_layers',type=int,default=3)
training_group.add_argument('--lr_actor', type=float, default=0.0001)
training_group.add_argument('--lr_critic', type=float, default=0.0001)
training_group.add_argument('--lr_alpha', type=float, default=0.0001)

training_group.add_argument('--SAC_batch_size',type=int,default=4096)
training_group.add_argument('--max_grad_norm', type=float, default=1.0)
training_group.add_argument('--num_training_step',type=int,default=int(1e7))
training_group.add_argument('--eval_interval',type=int,default=int(2e4))
training_group.add_argument('--num_eval_episodes',type=int,default=100)
training_group.add_argument('--max_episode_length',type=int,default=1000)
training_group.add_argument('--reward_factor',type=float,default=1.0)
training_group.add_argument('--tau',type=float,default=1e-2)
training_group.add_argument('--start_steps',type=int,default=int(5e3))
training_group.add_argument('--log_freq',type=int,default=int(1e4))
training_group.add_argument('--weight_path', type=str, default='./weights')

training_group.add_argument('--begin_cpu',type=int,default=0)
training_group.add_argument('--end_cpu',type=int,default=96)
training_group.add_argument('--num_envs',type=int,default=1)
training_group.add_argument('--eval_num_envs',type=int,default=25)

#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()
gamma                                   = args.gamma
device                                  = args.device_name
seed                                    = args.seed
buffer_size                             = args.buffer_size
SAC_batch_size                          = args.SAC_batch_size

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
lr_alpha                                = args.lr_alpha
max_grad_norm                           = args.max_grad_norm
num_training_step                       = args.num_training_step
eval_interval                           = args.eval_interval
num_eval_episodes                       = args.num_eval_episodes
env_name                                = args.env_name
reward_factor                           = args.reward_factor
tau                                     = args.tau
max_episode_length                      = args.max_episode_length
num_envs                                = args.num_envs
start_steps                             = args.start_steps
log_freq                                = args.log_freq
eval_num_envs                           = args.eval_num_envs
weight_path                             = args.weight_path

os.makedirs(weight_path,exist_ok=True)
weight_path = os.path.join(weight_path,env_name)
os.makedirs(weight_path,exist_ok=True)
weight_path = os.path.join(weight_path,'SAC')
log_path = f'{weight_path}/log_data'
os.makedirs(weight_path,exist_ok=True)
os.makedirs(log_path,exist_ok=True)

eval_return = open(f'{log_path}/return_{seed}.txt','w')
