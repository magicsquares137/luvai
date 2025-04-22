# config for supervised fine-tuning GPT-2 on No Robots dataset
wandb_log = True
wandb_project = 'no-robots-sft'
wandb_run_name = 'gpt2-124M-sft'

# Batch size settings
batch_size = 8  # Smaller batch size for fine-tuning
block_size = 1024
gradient_accumulation_steps = 8  # Same as before

# Training duration - shorter due to smaller dataset
max_iters = 86000  # Reduced from 20000
lr_decay_iters = 86000  # Match max_iters

# Learning rate settings
learning_rate = 5e-6  # Slightly lower than before
min_lr = 5e-7  # Maintain the same ratio

# Shorter warmup for smaller dataset
warmup_iters = 300  # Reduced from 1000

# More frequent evaluation for smaller dataset
eval_interval = 200  # More frequent than 500
eval_iters = 50  # Fewer evaluation iterations
log_interval = 10  # Keep the same

# Dataset setting
dataset = 'no_robots'  # Point to your processed dataset directory

# Weight decay - same as before
weight_decay = 0.1

# Dropout - keep the same
dropout = 0.1

# Resume from your pretrained model
init_from = 'resume'
init_resume_path = 'out'  # Path to your pretrained model checkpoint