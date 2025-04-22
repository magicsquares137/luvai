# config for supervised fine-tuning GPT-2 on tiny-codes dataset
wandb_log = True
wandb_project = 'tiny-codes-sft'
wandb_run_name = 'gpt2-124M-sft'

# Batch size settings
batch_size = 8  # Smaller batch size for fine-tuning
block_size = 1024
gradient_accumulation_steps = 9  # Reduced from pretraining

# Training duration - much shorter than pretraining
max_iters = 20000
lr_decay_iters = 20000

# Learning rate settings - lower for fine-tuning
learning_rate = 1e-5
min_lr = 1e-6

# Longer warmup for fine-tuning
warmup_iters = 1000

# Evaluation settings
eval_interval = 500
eval_iters = 100
log_interval = 10

# Dataset setting
dataset = 'tiny_codes_sft'  # Point to your processed dataset directory

# Weight decay - slightly reduced for fine-tuning
weight_decay = 0.1

# Dropout - add a small amount for fine-tuning
dropout = 0.1

# Resume from your pretrained model
init_from = 'resume'
init_resume_path = 'out'  # Path to your pretrained model checkpoint