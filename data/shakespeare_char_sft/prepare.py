"""
Prepare the Shakespeare conversations dataset for character-level language modeling.
Uses the same encoding as the original Shakespeare dataset.
Splits by example first, then tokenizes to maintain conversation integrity.
"""
import os
import pickle
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)

# Load the meta information from the original Shakespeare dataset
meta_path = os.path.join('../shakespeare_char', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

# Extract the character-to-integer mapping
stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

# Define encoding and decoding functions
def encode(s):
    return [stoi[c] for c in s if c in stoi]  # Only encode characters in our vocabulary

def decode(l):
    return ''.join([itos[i] for i in l])

# Load conversations from file
conversations_path = 'conversations.txt'
if not os.path.exists(conversations_path):
    print(f"Error: {conversations_path} not found.")
    print("Please create this file with your conversation examples.")
    exit(1)

with open(conversations_path, 'r', encoding='utf-8') as f:
    content = f.read().strip()

# Split into individual conversation examples
# Each example is separated by a blank line
conversations = content.split('\n\n')
print(f"Loaded {len(conversations)} conversation examples")

# Check for any issues in the examples
for i, conv in enumerate(conversations):
    if not conv.strip():
        print(f"Warning: Empty conversation at index {i}")
        continue
    
    if "Human:" not in conv or "Assistant:" not in conv:
        print(f"Warning: Conversation at index {i} missing Human: or Assistant:")

# Shuffle and split the conversations into train/val BEFORE tokenization
random.shuffle(conversations)
split_idx = int(len(conversations) * 0.9)  # 90% train, 10% val
train_conversations = conversations[:split_idx]
val_conversations = conversations[split_idx:]

print(f"Split into {len(train_conversations)} training and {len(val_conversations)} validation examples")

# Function to process and clean a set of conversations
def process_conversations(convs):
    all_text = '\n\n'.join(convs)
    
    # Check for any characters not in the original vocabulary
    unique_chars = set(all_text)
    unknown_chars = [c for c in unique_chars if c not in stoi]
    if unknown_chars:
        print(f"Warning: Found {len(unknown_chars)} characters not in original vocabulary: {''.join(unknown_chars)}")
        
        # Fix the text by replacing unknown characters
        for c in unknown_chars:
            if c == '"':  # Common quotation mark
                all_text = all_text.replace(c, "'")
            elif c == '"':  # Common quotation mark
                all_text = all_text.replace(c, "'")
            elif c == '-':  # Ellipsis
                all_text = all_text.replace(c, ",")
            else:
                all_text = all_text.replace(c, "")
    
    # Encode the text
    return encode(all_text)

# Process train and val sets
train_ids = process_conversations(train_conversations)
val_ids = process_conversations(val_conversations)

print(f"Train tokens: {len(train_ids):,}")
print(f"Validation tokens: {len(val_ids):,}")

# Convert to numpy arrays of uint16
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

# Save to binary files
train_path = os.path.join('.', 'train.bin')
val_path = os.path.join('.', 'val.bin')
train_ids.tofile(train_path)
val_ids.tofile(val_path)

print(f"Saved {train_path} and {val_path}")

# Also save our own copy of meta.pkl for consistency
meta_out_path = os.path.join('.', 'meta.pkl')
with open(meta_out_path, 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved metadata to {meta_out_path}")
print("Preparation complete!")