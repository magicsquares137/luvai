import os
import torch
import tiktoken
from model import GPTConfig, GPT

# Configuration
checkpoint_path = "out/ckpt.pt"  # Path to your SFT checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 200  # Maximum length of generated response
temperature = 0.8  # Higher = more creative, lower = more deterministic
top_k = 40  # Limit to top K token choices for each step

# Setup tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|im_start|>", "<|im_end|>"})
decode = lambda l: enc.decode(l)

# Load model
def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    # Fix the keys in the state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model

# Generate text
def generate_response(model, prompt):
    # Format the prompt with chat markers
    if not prompt.startswith("<|im_start|>user"):
        prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
    
    # Encode and create tensor
    prompt_ids = encode(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode full output
    full_output = decode(y[0].tolist())
    
    # Extract only the assistant's response
    response = full_output[len(prompt):]
    
    # Clean up response if needed
    if not response.startswith("<|im_start|>assistant"):
        response = f"<|im_start|>assistant\n{response}"
    
    # Remove any generation beyond the assistant's turn
    if "<|im_start|>user" in response:
        response = response.split("<|im_start|>user")[0].strip()
    
    # Make sure it ends properly
    if not response.endswith("<|im_end|>"):
        response = response + "<|im_end|>"
    
    # Return cleaned response
    return response

# Interactive loop
def interactive_demo():
    print("Loading model from checkpoint...")
    model = load_model(checkpoint_path)
    print("Model loaded! Enter prompts to test (type 'exit' to quit).")
    
    while True:
        user_input = input("\nUser prompt: ")
        if user_input.lower() == 'exit':
            break
        
        print("\nGenerating response...")
        response = generate_response(model, user_input)
        
        # Clean for display
        clean_response = response.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "")
        print(f"\nModel response:\n{clean_response}")

if __name__ == "__main__":
    interactive_demo()