"""
Improved sampling script for conversation-based models
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-rlhf' # your fine-tuned model directory
start = "Human: " # Start with a human prompt
num_samples = 1 # Just generate one sample for clarity
max_new_tokens = 500 # number of tokens generated
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 40 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # PyTorch 2.0 compilation
stop_after_assistant = True # Stop generation after Assistant's turn

# exec(open('configurator.py').read()) # Uncomment if you want command line overrides
# -----------------------------------------------------------------------------

# Set up the environment
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load the model
ckpt_path = os.path.join(out_dir, 'ckpt_best.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# Get the encoding/decoding functions from meta.pkl
meta_path = os.path.join('data', 'shakespeare_char_sft', 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("Error: No meta.pkl found for character encoding")
    exit(1)

# Get the starting prompt from user
if not start.startswith("Human:"):
    start = "Human: " + start
human_prompt = input(f"Enter your question (or press Enter to use default '{start}'): ")
if human_prompt.strip():
    start = "Human: " + human_prompt

print("\n--- Generating response ---\n")
print(start)

# Encode the prompt
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Generate token by token with early stopping for conversation format
with torch.no_grad():
    with ctx:
        # Initial generation
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        output = decode(y[0].tolist())
        
        # If we want to stop after "Assistant:" turn, look for the next "Human:" and truncate
        if stop_after_assistant and "Human:" in output[len(start):]:
            output = output[:output.find("Human:", len(start))]
        
        print(output)

print("\n--- End of generation ---\n")