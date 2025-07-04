# config for supervised fine-tuning on chat dataset
wandb_log = True
wandb_project = 'chat-sft'
wandb_run_name = 'gpt2-124M-chat-sft'

# Batch size settings
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 9  # Divisible by 3 for 3 GPUs

# Training duration
max_iters = 73000  # Adjust based on dataset size
lr_decay_iters = 73000

# Learning rate settings
learning_rate = 5e-6
min_lr = 5e-7

# Warmup
warmup_iters = 200

# Evaluation settings
eval_interval = 100
eval_iters = 50
log_interval = 10

# Dataset setting
dataset = 'chat_finetuning'

# Weight decay
weight_decay = 0.1

# Dropout
dropout = 0.1

# Resume from your pretrained model
init_from = 'resume'
init_resume_path = 'out'  # Path to your pretrained model checkpoint