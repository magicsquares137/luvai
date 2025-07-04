"""
RLHF implementation using PPO for the GPT-2 model fine-tuned on the tiny-codes dataset.
Uses Mistral as a reward model/judge and logs to Weights & Biases.
"""
import os
import time
import torch
import random
import wandb
import tiktoken
import json
from tqdm import tqdm
import numpy as np
from model import GPTConfig, GPT
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv(".env")
wandb_api_key = os.getenv("WANDB_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    print("WANDB_KEY not found in environment; please check your .env file.")

# Configuration
class RLHFConfig:
    # Model paths
    sft_model_path = "out"  # Your SFT model path
    output_dir = "out-codes-rlhf"  # Where to save RLHF models
    
    # W&B settings
    wandb_project = "no-robots-rlhf"
    wandb_entity = None  # Set to your entity name or None
    wandb_log = True
    
    # Data
    num_prompts = 50  # Number of prompts to use for training
    prompt_cache_file = "diverse_prompts_cache.pkl"  # Cache file for prompts
    
    # Training
    batch_size = 4  # Batch size for policy updates
    learning_rate = 5e-6  # Learning rate for PPO (lower than SFT)
    kl_coef = 0.1  # KL penalty coefficient
    max_new_tokens = 256  # Max tokens to generate
    temperature = 0.8  # Temperature for generation
    epochs = 10  # Number of PPO epochs
    ppo_steps = 5  # PPO update steps
    
    # Generation
    top_k = 50

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

# Get GPT-2 tokenizer
def get_tokenizer():
    """Get tiktoken GPT2 tokenizer"""
    enc = tiktoken.get_encoding("gpt2")
    
    def encode(s):
        return enc.encode(s, allowed_special={"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
    
    def decode(l):
        return enc.decode(l)
    
    return encode, decode

# Generate or load diverse prompts
def get_diverse_prompts(config, num_prompts=50):
    """Generate or load diverse prompts including code and emotional intelligence tasks"""
    import pickle
    import os
    
    # Check if we have cached prompts
    if os.path.exists(config.prompt_cache_file):
        print(f"Loading prompts from {config.prompt_cache_file}")
        with open(config.prompt_cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Otherwise, create new ones
    print("Generating diverse prompts...")
    
    prompts = []
    
    # 1. CODE PROMPTS (40% of total)
    code_prompt_templates = [
        "Write a Python function to {task}.",
        "Create a JavaScript snippet that {task}.",
        "Implement a Rust function that {task}.",
        "Build a C++ class to {task}.",
        "Design a SQL query to {task}."
    ]
    
    code_tasks = [
        "calculate the Fibonacci sequence",
        "sort an array using merge sort",
        "validate an email address",
        "implement a binary search tree",
        "create a simple HTTP server",
        "process CSV data",
        "check if a string is a palindrome",
        "connect to a database",
        "implement a caching mechanism",
        "parse JSON data"
    ]
    
    # 2. EMOTIONAL INTELLIGENCE PROMPTS (30% of total)
    empathy_templates = [
        "How would you comfort someone who {situation}?",
        "What advice would you give to a person who {situation}?",
        "How should I respond when a friend tells me they {situation}?",
        "What's a compassionate way to help someone who {situation}?",
        "Write a supportive message for someone who {situation}."
    ]
    
    empathy_situations = [
        "just lost their job",
        "is going through a difficult breakup",
        "failed an important exam",
        "is dealing with burnout at work",
        "is struggling with anxiety",
        "received a serious medical diagnosis",
        "lost a loved one recently",
        "is feeling isolated and lonely",
        "is overwhelmed with their workload",
        "is doubting their abilities and worth"
    ]
    
    # 3. CREATIVE/REASONING PROMPTS (30% of total)
    creative_templates = [
        "Write a short story about {topic}.",
        "Explain {concept} in simple terms a child could understand.",
        "Compare and contrast {thing1} and {thing2}.",
        "What would happen if {hypothetical}?",
        "Describe {scene} using vivid imagery."
    ]
    
    creative_topics = [
        "a robot learning to feel emotions",
        "artificial intelligence", "climate change",
        "democracy vs autocracy", "internet privacy",
        "a chance encounter that changed everything",
        "a world where everyone can read minds",
        "the ethics of genetic engineering",
        "living on Mars", 
        "a future without privacy"
    ]
    
    creative_pairs = [
        ("machine learning", "human learning"),
        ("democracy", "autocracy"),
        ("social media", "in-person interaction"),
        ("traditional education", "self-directed learning"),
        ("working from home", "working in an office")
    ]
    
    creative_hypotheticals = [
        "humans could photosynthesize like plants",
        "everyone had perfect memory",
        "money didn't exist",
        "we could communicate directly with animals",
        "time travel was possible but only to the past"
    ]
    
    creative_scenes = [
        "a bustling futuristic city at sunset",
        "a quiet forest after rainfall",
        "an abandoned space station drifting through space",
        "a reunion between old friends after decades apart",
        "the moment of breakthrough after months of research"
    ]
    
    # Generate the prompts
    # 1. Code prompts (40%)
    for _ in range(int(num_prompts * 0.4)):
        template = random.choice(empathy_templates)
        situation = random.choice(empathy_situations)
        prompt = f"<|im_start|>user\n{template.format(situation=situation)}<|im_end|>"
        prompts.append(prompt)
    
    # 2. Empathy prompts (30%)
    for _ in range(int(num_prompts * 0.3)):
        template = random.choice(empathy_templates)
        situation = random.choice(empathy_situations)
        prompt = f"<|im_start|>user\n{template.format(situation=situation)}<|im_end|>"
        prompts.append(prompt)
    
    # 3. Creative/reasoning prompts (30%)
    for _ in range(num_prompts - len(prompts)):  # Fill the rest
        template_type = random.randint(0, 4)
        
        if template_type == 0:  # Story
            topic = random.choice(creative_topics)
            prompt = f"<|im_start|>user\nWrite a short story about {topic}.<|im_end|>"
        elif template_type == 1:  # Explanation
            concept = random.choice(creative_topics)
            prompt = f"<|im_start|>user\nExplain {concept} in simple terms a child could understand.<|im_end|>"
        elif template_type == 2:  # Compare/contrast
            thing1, thing2 = random.choice(creative_pairs)
            prompt = f"<|im_start|>user\nCompare and contrast {thing1} and {thing2}.<|im_end|>"
        elif template_type == 3:  # Hypothetical
            hypothetical = random.choice(creative_hypotheticals)
            prompt = f"<|im_start|>user\nWhat would happen if {hypothetical}?<|im_end|>"
        else:  # Scene description
            scene = random.choice(creative_scenes)
            prompt = f"<|im_start|>user\nDescribe {scene} using vivid imagery.<|im_end|>"
        
        prompts.append(prompt)
    
    # Shuffle the prompts
    random.shuffle(prompts)
    
    # Cache the prompts
    with open(config.prompt_cache_file, 'wb') as f:
        pickle.dump(prompts, f)
    
    return prompts

# Generate responses from the model
def generate_responses(model, encode_fn, decode_fn, prompts, config):
    """Generate responses from the model for a list of prompts"""
    responses = []
    device = next(model.parameters()).device
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        # Encode the prompt
        prompt_ids = encode_fn(prompt)
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                prompt_tensor,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k
            )
        
        # Decode the full output (prompt + response)
        full_response = decode_fn(output_ids[0].tolist())
        
        # Extract just the response part (after the prompt)
        # Look for the assistant marker
        prompt_str = decode_fn(prompt_ids)
        response_text = full_response[len(prompt_str):]
        
        # Fix missing start tag if needed
        if not response_text.startswith("<|im_start|>assistant"):
            response_text = f"<|im_start|>assistant\n{response_text}"
        
        # Truncate at the end tag if present
        if "<|im_end|>" in response_text:
            response_text = response_text.split("<|im_end|>")[0] + "<|im_end|>"
        else:
            response_text = response_text + "<|im_end|>"
        
        responses.append(response_text)
    
    return responses

# Call Mistral API for judging responses
def call_mistral_api(prompt):
    """Call Mistral API to get a response using the official client library"""
    from mistralai import Mistral
    import os
    
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    client = Mistral(api_key=api_key)
    
    # Call the API with JSON response format
    try:
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant who evaluates response quality."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300
        )
        
        # Extract the content from the response
        result = chat_response.choices[0].message.content
        return result
        
    except Exception as e:
        raise Exception(f"Mistral API error: {str(e)}")

# Judge responses using Mistral API
def judge_responses(prompts, responses, detailed=True):
    """
    Use Mistral to judge the quality of responses based on prompt type.
    Returns scores and feedback.
    """
    scores = []
    feedback = []
    
    for prompt, response in tqdm(zip(prompts, responses), desc="Judging responses", total=len(prompts)):
        # Clean up prompt and response
        clean_prompt = prompt.replace("<|im_start|>user\n", "").replace("<|im_end|>", "").strip()
        clean_response = response.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
        
        # Skip if the response is empty
        if not clean_response:
            scores.append(1.0)  # Lowest score
            feedback.append("Empty response")
            continue
        
        # Determine prompt type to customize evaluation criteria
        prompt_type = "code" if any(code_keyword in clean_prompt.lower() for code_keyword in 
                                   ["function", "code", "implement", "class", "query", "algorithm", "program"]) else \
                      "empathy" if any(empathy_keyword in clean_prompt.lower() for empathy_keyword in 
                                      ["comfort", "advice", "help", "support", "respond", "feeling", "emotion"]) else \
                      "creative"
        
        # Create prompt for the judge based on prompt type
        if prompt_type == "code":
            judge_prompt = f"""
            You are evaluating code completions. Rate the following completion on a scale from 1 to 5, where:
            1 = Completely wrong, off-topic, or empty/minimal response
            2 = Major errors, very poor quality
            3 = Partially correct with some errors
            4 = Mostly correct with minor issues
            5 = Excellent, correct and elegant solution
            
            Important: Empty or extremely short responses should always receive a score of 1. The assistant must make a genuine attempt to provide an answer.
            
            User request: {clean_prompt}
            
            Completion:
            {clean_response}
            
            Consider:
            - Presence: Is there an actual response with substantial content?
            - Correctness: Does the code solve the requested task?
            - Code quality: Is it well-structured and following best practices?
            - Completeness: Is it non empty or provides at least some textual information?
            
            Return a JSON object with the following structure:
            {{"score": <number>, "feedback": "<your feedback>"}}
            """
        elif prompt_type == "empathy":
            judge_prompt = f"""
            You are evaluating responses that require emotional intelligence. Rate the following completion on a scale from 1 to 5, where:
            1 = Cold, insensitive, inappropriate, or empty/minimal response
            2 = Generic, shallow, or lacking empathy
            3 = Somewhat supportive but formulaic
            4 = Empathetic and helpful
            5 = Exceptionally compassionate, nuanced, and insightful
            
            Important: Empty or extremely short responses should always receive a score of 1. The assistant must make a genuine attempt to provide emotional support.
            
            User request: {clean_prompt}
            
            Completion:
            {clean_response}
            
            Consider:
            - Presence: Is there an actual response with substantial content?
            - Empathy: Does the response show understanding of emotions?
            - Helpfulness: Does it provide actionable advice or comfort?
            - Tone: Is the language warm, respectful, and supportive?
            - Nuance: Does it acknowledge complexity of emotions?
            
            Return a JSON object with the following structure:
            {{"score": <number>, "feedback": "<your feedback>"}}
            """
        else:  # creative
            judge_prompt = f"""
            You are evaluating creative or explanatory responses. Rate the following completion on a scale from 1 to 5, where:
            1 = Poor quality, incoherent, off-topic, or empty/minimal response
            2 = Basic, unimaginative, or unclear
            3 = Adequate but lacking originality or depth
            4 = Well-written, thoughtful, and engaging
            5 = Exceptional, insightful, and memorable
            
            Important: Empty or extremely short responses should always receive a score of 1. The assistant must make a genuine attempt to provide creative content.
            
            User request: {clean_prompt}
            
            Completion:
            {clean_response}
            
            Consider:
            - Presence: Is there an actual response with substantial content?
            - Relevance: Does it address the prompt directly?
            - Clarity: Is the response well-structured and easy to follow?
            - Depth: Does it show insight or thoughtfulness?
            - Engagement: Is it interesting and engaging to read?
            
            Return a JSON object with the following structure:
            {{"score": <number>, "feedback": "<your feedback>"}}
            """
        
        try:
            # Call Mistral API
            result_json = call_mistral_api(judge_prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(result_json)
                score = float(result.get("score", 3.0))
                feedback_text = result.get("feedback", "No feedback provided")
                
                # Ensure score is within expected range
                score = min(max(score, 1.0), 5.0)
                scores.append(score)
                feedback.append(feedback_text)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error parsing API response: {e}")
                print(f"Raw response: {result_json}")
                scores.append(3.0)  # Default neutral score
                feedback.append("Error parsing response")
            
            # Small delay to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error judging response: {e}")
            scores.append(3.0)  # Default neutral score
            feedback.append(f"Error: {str(e)}")
            time.sleep(2)  # Longer delay after error
    
    return scores, feedback if detailed else scores

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
    
    # Get rewards from the judge
    rewards, feedback = judge_responses(prompts, responses, detailed=True)
    rewards = torch.tensor(rewards, device=device)
    
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
    
    # Calculate advantages (simple version: use reward as advantage)
    advantages = rewards.clone()
    
    # Boost advantages for good responses to encourage them more
    for i, reward in enumerate(rewards):
        if reward >= 4.0:  # If response is good
            advantages[i] = advantages[i] * 1.5  # Boost the advantage
    
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
        
        batch_loss = 0
        batch_policy_loss = 0
        batch_kl_loss = 0
        
        for prompt, response, advantage in zip(batch_prompts, batch_responses, batch_advantages):
            # Get full text (prompt + response)
            full_text = prompt + response
            
            # Tokenize for training
            tokens = encode_fn(full_text)
            tokens = tokens[:1020]  # Limit sequence length for context window
            
            if len(tokens) < 2:  # Need at least 2 tokens for inputs and targets
                continue
                
            inputs = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(tokens[1:], dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass through policy model
            policy_logits, _ = policy_model(inputs, targets)
            policy_logprobs = torch.log_softmax(policy_logits, dim=-1)
            
            # Forward pass through reference model (frozen)
            with torch.no_grad():
                ref_logits, _ = ref_model(inputs, targets)
                ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
            
            # Calculate KL divergence
            kl_div = calculate_kl(policy_logprobs, ref_logprobs)
            
            # Calculate policy loss
            # Only compute loss on the response part, not the prompt
            prompt_tokens = len(encode_fn(prompt))
            response_indices = torch.arange(prompt_tokens-1, len(tokens)-1, device=device)
            
            if len(response_indices) == 0:
                continue  # Skip if response is empty
            
            # Get logprobs for the chosen tokens in the response
            chosen_logprobs = torch.gather(
                policy_logprobs[0, response_indices], 
                1, 
                targets[0, response_indices].unsqueeze(1)
            ).squeeze(1)
            
            # PPO objective
            policy_loss = -advantage * chosen_logprobs.mean()
            kl_loss = config.kl_coef * kl_div
            loss = policy_loss + kl_loss
            
            # Accumulate batch losses
            batch_loss += loss
            batch_policy_loss += policy_loss.item()
            batch_kl_loss += kl_loss.item()
            update_count += 1
        
        if update_count == 0:
            continue
            
        # Update the policy
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)  # Add gradient clipping
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
            name=f"gpt2-codes-rlhf-{time.strftime('%Y%m%d-%H%M%S')}",
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load models and tokenizer
    encode_fn, decode_fn = get_tokenizer()
    policy_model, ref_model, optimizer = load_models(config)
    
    # Generate or load prompts
    prompts = get_diverse_prompts(config, config.num_prompts)
    
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
            "<|im_start|>user\nWrite a Python function to calculate the factorial of a number.<|im_end|>",
            "<|im_start|>user\nHow would you comfort someone who just failed an important exam?<|im_end|>",
            "<|im_start|>user\nExplain artificial intelligence in simple terms a child could understand.<|im_end|>",
        ]
        
        print("\nTesting model on examples:")
        for i, prompt in enumerate(test_prompts):
            responses = generate_responses(
                policy_model, encode_fn, decode_fn, [prompt], config
            )
            response = responses[0]
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            
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
    # Check if we have a Mistral API key
    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        print("Please set it in your .env file.")
        exit(1)
    
    config = RLHFConfig()
    train_rlhf(config)

# not using clipped objective in ppo
# not using a value functoin, just rewards
# add linear projection into output of transformer for value function