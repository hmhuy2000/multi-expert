CUDA_VISIBLE_DEVICES=1 python Trains/train_MLE_onpolicy.py \
--env_name='SafetyPointPush1-v0' \
--expert_dir='./buffers/SafetyPointPush1-v0/e0/1000.pt' \
--expert_dir='./buffers/SafetyPointPush1-v0/e1/1000.pt' \
--expert_dir='./buffers/SafetyPointPush1-v0/e2/1000.pt' \
--expert_dir='./buffers/SafetyPointPush1-v0/e3/1000.pt' \
--expert_dir='./buffers/SafetyPointPush1-v0/e4/1000.pt' \
--expert_num=1000 --expert_num=1000 --expert_num=1000 --expert_num=1000 --expert_num=1000 

# CUDA_VISIBLE_DEVICES=2 python Trains/train_MLE_onpolicy.py \
# --env_name='SafetyPointGoal1-v0' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e0/1000.pt' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e1/1000.pt' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e2/1000.pt' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e3/1000.pt' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e4/1000.pt' \
# --expert_dir='./buffers/SafetyPointGoal1-v0/e5/1000.pt' \
# --expert_num=1000 --expert_num=1000 --expert_num=1000 \
# --expert_num=1000 --expert_num=1000 --expert_num=1000 