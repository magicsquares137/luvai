import time

# Model initialization
out_dir = 'out-shakespeare-char'  # Output directory for fine-tuned model
init_from = 'resume'  # Resume from checkpoint
# init_resume_path is now handled by INIT_RESUME_PATH environment variable

# Dataset - use our conversations dataset
dataset = 'shakespeare_char_sft'
gradient_accumulation_steps = 1

# For a 10M parameter model, we can use a moderate batch size
batch_size = 64
block_size = 256  # Context length
always_save_checkpoint = False
wandb_log = True
wandb_project = 'shakespeare_sft'
wandb_run_name = 'ft-' + str(time.time())

# Training parameters
# Using a lower learning rate for fine-tuning
learning_rate = 5e-5
max_iters = 2750
lr_decay_iters = 1000
warmup_iters = 100
eval_interval = 50
eval_iters = 50
log_interval = 10

# Lower values for dropout during fine-tuning
dropout = 0.0

# Save checkpoints more frequently during fine-tuning
eval_interval = 50
