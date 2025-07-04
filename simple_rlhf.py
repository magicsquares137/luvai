"""
A simple RLHF implementation using PPO for the Shakespeare model.
Uses Mistral as a reward model/judge and logs to Weights & Biases.
"""
import os
import pickle
import time
import torch
import random
import wandb
from tqdm import tqdm
import numpy as np
from utils import judge_responses, generate_shakespeare_prompts
from model import GPTConfig, GPT
from dotenv import load_dotenv

load_dotenv(".env")

wandb_api_key = os.getenv("WANDB_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    print("WANDB_KEY not found in environment; please check your .env_dev file.")

# Configuration
class RLHFConfig:
    # Model paths
    sft_model_path = "out-shakespeare-char"  # Your SFT model path
    output_dir = "out-shakespeare-rlhf"  # Where to save RLHF models
    
    # W&B settings
    wandb_project = "shakespeare-rlhf"
    wandb_entity = None  # Set to your entity name or None
    wandb_log = True
    
    # Data
    num_prompts = 50  # Number of prompts to use for training
    prompt_cache_file = "prompts_cache.pkl"  # Cache file for prompts
    
    # Training
    batch_size = 4  # Batch size for policy updates
    learning_rate = 1e-5  # Learning rate for PPO
    kl_coef = 0.35  # KL penalty coefficient
    max_new_tokens = 150  # Max tokens to generate
    temperature = 0.8  # Temperature for generation
    epochs = 10  # Number of PPO epochs
    ppo_steps = 10  # PPO update steps
    
    # Generation
    top_k = 40

# Utility function to load the character-level encoding
def load_encoding_functions(model_path):
    """Load the encoding functions from the meta.pkl file"""
    meta_path = os.path.join('data', 'shakespeare_char_sft', 'meta.pkl')
    
    if not os.path.exists(meta_path):
        raise ValueError(f"Meta file not found at {meta_path}")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    def encode(s):
        return [stoi[c] for c in s if c in stoi]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    return encode, decode

# Simple tokenizer wrapper for compatibility
class CharTokenizer:
    def __init__(self, encode_fn, decode_fn):
        self.encode = encode_fn
        self.decode = decode_fn
    
    def __call__(self, text):
        return self.encode(text)

# Load models
def load_models(config):
    """Load the policy model and reference model"""
    # Load the checkpoint
    ckpt_path = os.path.join(config.sft_model_path, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the policy model
    gptconf = GPTConfig(**checkpoint['model_args'])
    policy_model = GPT(gptconf)
    
    # Fix the keys in the state dict by removing the "_orig_mod." prefix
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load the cleaned state dict
    policy_model.load_state_dict(state_dict)
    policy_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model.eval()  # Set to eval mode for generation
    
    # Create a frozen reference model (copy of the policy model)
    ref_model = GPT(gptconf)
    ref_model.load_state_dict(state_dict)  # Use the already cleaned state dict
    ref_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    ref_model.eval()
    
    # Make reference model's parameters non-trainable
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Create optimizer for policy model
    optimizer = torch.optim.Adam(
        policy_model.parameters(),
        lr=config.learning_rate,
    )
    
    return policy_model, ref_model, optimizer

def generate_responses(model, encode_fn, decode_fn, prompts, config):
    """Generate responses from the model for a list of prompts"""
    responses = []
    device = next(model.parameters()).device
    
    for prompt in prompts:
        valid_response = False
        current_temp = config.temperature
        max_attempts = 3
        
        # Try up to max_attempts times with increasing temperature
        for attempt in range(max_attempts):
            # Encode the prompt
            prompt_ids = encode_fn(prompt)
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # Generate response with current temperature
            with torch.no_grad():
                output_ids = model.generate(
                    prompt_tensor,
                    max_new_tokens=config.max_new_tokens,
                    temperature=current_temp,
                    top_k=config.top_k
                )
            
            # Decode the full output (prompt + response)
            full_response = decode_fn(output_ids[0].tolist())
            
            # Only keep the response part (after the prompt)
            response = full_response[len(prompt):]
            
            # Clean up - ensure it starts with "Assistant:"
            if not response.strip().startswith("Assistant:"):
                response = "Assistant: " + response.strip()
            
            # Remove any Human: parts that might have been generated
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            
            # Check if we have a valid response (more than just "Assistant:")
            if len(response.strip()) > 10 and response.strip() != "Assistant:":
                valid_response = True
                break
            
            # Increase temperature for next attempt
            current_temp *= 1.5
        
        # If we still don't have a valid response after all attempts,
        # create a default response based on the prompt
        if not valid_response:
            topic = prompt.split(":", 1)[1].strip() if ":" in prompt else "this query"
            response = f"Assistant: On the matter of {topic}, I would say that it resembles a gentle wind that stirs the soul to contemplation."
        
        responses.append(response)
    
    return responses

# Calculate KL divergence
def calculate_kl(policy_logprobs, ref_logprobs):
    """Calculate KL divergence between policy and reference model logprobs"""
    return (torch.exp(policy_logprobs) * (policy_logprobs - ref_logprobs)).mean()

# PPO training step
def ppo_step(policy_model, ref_model, optimizer, encode_fn, decode_fn, prompts, config, epoch):
    """Perform one PPO training step"""
    device = next(policy_model.parameters()).device
    
    # Generate responses with the current policy
    responses = generate_responses(policy_model, encode_fn, decode_fn, prompts, config)

    # Filter out empty responses and their prompts
    filtered_prompts = []
    filtered_responses = []
    
    for prompt, response in zip(prompts, responses):
        if len(response.strip()) > 10 and not response.strip() == "Assistant:":
            filtered_prompts.append(prompt)
            filtered_responses.append(response)
    
    # Log how many were filtered
    if config.wandb_log:
        wandb.log({"filtered_prompts": len(prompts) - len(filtered_prompts)})
    
    # Skip this step if all responses were empty
    if len(filtered_prompts) == 0:
        print("Warning: All responses were empty for this batch")
        return {'rewards/mean': 0, 'rewards/min': 0, 'rewards/max': 0, 'rewards/std': 0}
    
    # Get rewards from the judge
    rewards, feedback = judge_responses(prompts, responses, detailed=True)
    rewards = torch.tensor(rewards, device=device)

    # Penalize empty responses
    for i, response in enumerate(responses):
        if not response.strip() or response.strip() == "Assistant:":
            rewards[i] = 1.0  # Lowest possible score
            if config.wandb_log:
                wandb.log({"empty_responses": 1})
            # Optionally update feedback
            feedback[i] = "Empty or minimal response penalized"
    
    # Log some sample responses and rewards to W&B
    if config.wandb_log:
        sample_idx = random.sample(range(len(prompts)), min(3, len(prompts)))
        for i in sample_idx:
            wandb.log({
                f"examples/prompt_{i}": prompts[i],
                f"examples/response_{i}": responses[i],
                f"examples/reward_{i}": rewards[i].item(),
                f"examples/feedback_{i}": feedback[i],
            })
    
    # Calculate advantages
    # For simplicity, we'll just use the reward as the advantage
    advantages = rewards.clone()
    for i, reward in enumerate(rewards):
        if reward >= 4.0:  # If response is good
            advantages[i] = advantages[i] * 2.0  # Double the advantage
    
    # Track metrics for this epoch
    metrics = {
        'rewards/mean': rewards.mean().item(),
        'rewards/min': rewards.min().item(),
        'rewards/max': rewards.max().item(),
        'rewards/std': rewards.std().item(),
    }
    
    # PPO update loop
    total_loss = 0
    total_policy_loss = 0
    total_kl_loss = 0
    update_count = 0
    
    for step in range(config.ppo_steps):
        # Sample a batch of prompts for update
        batch_indices = random.sample(range(len(prompts)), min(config.batch_size, len(prompts)))
        batch_prompts = [prompts[i] for i in batch_indices]
        batch_responses = [responses[i] for i in batch_indices]
        batch_advantages = advantages[batch_indices]
        
        # Convert to tensors
        batch_loss = 0
        batch_policy_loss = 0
        batch_kl_loss = 0
        
        for prompt, response, advantage in zip(batch_prompts, batch_responses, batch_advantages):
            # Get full text (prompt + response)
            full_text = prompt + response
            
            # Tokenize for training
            tokens = encode_fn(full_text)
            tokens = tokens[:512]  # Limit sequence length
            inputs = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass through policy model
            policy_logits, _ = policy_model(inputs, targets)
            policy_logprobs = torch.log_softmax(policy_logits, dim=-1)
            
            # Forward pass through reference model
            with torch.no_grad():
                ref_logits, _ = ref_model(inputs, targets)
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
            
            # Calculate token-level KL divergence
            kl_div = calculate_kl(policy_logprobs, ref_logprobs)
            
            # Calculate policy loss
            # Only compute loss on the response part, not the prompt
            prompt_tokens = len(encode_fn(prompt))
            response_indices = torch.arange(prompt_tokens, len(tokens)-1, device=device)
            if len(response_indices) == 0:
                continue  # Skip if response is empty
                
            # Get logprobs for the chosen tokens
            chosen_logprobs = torch.gather(
                policy_logprobs[0, response_indices], 
                1, 
                targets[0, response_indices].unsqueeze(1)
            ).squeeze(1)
            
            # PPO objective
            policy_loss = -advantage * chosen_logprobs.mean()
            kl_loss = config.kl_coef * kl_div
            loss = policy_loss + kl_loss
            
            batch_loss += loss
            batch_policy_loss += policy_loss.item()
            batch_kl_loss += kl_loss.item()
            update_count += 1
        
        if batch_loss == 0:
            continue
            
        # Update the policy
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        total_policy_loss += batch_policy_loss
        total_kl_loss += batch_kl_loss
    
    # Log additional metrics
    if update_count > 0:
        metrics.update({
            'loss/total': total_loss / update_count,
            'loss/policy': total_policy_loss / update_count,
            'loss/kl': total_kl_loss / update_count,
        })
    
    # Log to W&B
    if config.wandb_log:
        wandb.log(metrics)
    
    return metrics

# Main RLHF training function
def train_rlhf(config):
    """Main RLHF training loop"""
    print("Starting RLHF training...")
    
    # Initialize W&B
    if config.wandb_log:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config),
            name=f"shakespeare-rlhf-{time.strftime('%Y%m%d-%H%M%S')}",
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load encoding functions and models
    encode_fn, decode_fn = load_encoding_functions(config.sft_model_path)
    tokenizer = CharTokenizer(encode_fn, decode_fn)
    policy_model, ref_model, optimizer = load_models(config)
    
    # Generate or load prompts
    if os.path.exists(config.prompt_cache_file):
        print(f"Loading prompts from {config.prompt_cache_file}")
        with open(config.prompt_cache_file, 'rb') as f:
            prompts = pickle.load(f)
    else:
        print(f"Generating {config.num_prompts} prompts...")
        prompts = generate_shakespeare_prompts(config.num_prompts)
        
        # Save prompts for reproducibility
        with open(config.prompt_cache_file, 'wb') as f:
            pickle.dump(prompts, f)
    
    # RLHF training loop
    best_reward = 0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # Sample prompts for this epoch
        epoch_prompts = random.sample(prompts, min(len(prompts), 20))
        
        # Run PPO step
        stats = ppo_step(
            policy_model, ref_model, optimizer, 
            encode_fn, decode_fn, epoch_prompts, config, epoch
        )
        
        # Print stats
        print(f"Epoch {epoch+1} results:")
        print(f"  Mean reward: {stats['rewards/mean']:.4f}")
        print(f"  Min reward: {stats['rewards/min']:.4f}")
        print(f"  Max reward: {stats['rewards/max']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'model': policy_model.state_dict(),
            'model_args': policy_model.config.__dict__,
            'epoch': epoch,
            'stats': stats,
        }
        
        checkpoint_path = os.path.join(config.output_dir, f'ckpt_epoch_{epoch+1}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if stats['rewards/mean'] > best_reward:
            best_reward = stats['rewards/mean']
            best_path = os.path.join(config.output_dir, 'ckpt_best.pt')
            torch.save(checkpoint, best_path)
            print(f"  New best model saved with reward: {best_reward:.4f}")
            
            if config.wandb_log:
                wandb.run.summary["best_reward"] = best_reward
                wandb.run.summary["best_epoch"] = epoch + 1
        
        # Test the model on a few examples
        test_prompts = [
            "Human: What is the meaning of life?",
            "Human: Tell me about love and betrayal.",
        ]
        
        print("\nTesting model on examples:")
        example_responses = []
        for i, prompt in enumerate(test_prompts):
            responses = generate_responses(
                policy_model, encode_fn, decode_fn, [prompt], config
            )
            response = responses[0]
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            
            example_responses.append(response)
            
            # Log examples to W&B
            if config.wandb_log:
                wandb.log({
                    f"test_examples/prompt_{i}": prompt,
                    f"test_examples/response_{i}": response,
                })
    
    # Save final model
    final_checkpoint = {
        'model': policy_model.state_dict(),
        'model_args': policy_model.config.__dict__,
        'epochs': config.epochs,
        'final_stats': stats,
    }
    torch.save(final_checkpoint, os.path.join(config.output_dir, 'ckpt_final.pt'))
    
    print(f"\nRLHF training complete! Final model saved to {config.output_dir}/ckpt_final.pt")
    print(f"Best model saved to {config.output_dir}/ckpt_best.pt with reward: {best_reward:.4f}")
    
    # Finish W&B run
    if config.wandb_log:
        wandb.finish()
    
    return policy_model

if __name__ == "__main__":
    # Check for Mistral API key
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        print("Please set it with: export MISTRAL_API_KEY=your_api_key")
        exit(1)
    
    config = RLHFConfig()
    train_rlhf(config)