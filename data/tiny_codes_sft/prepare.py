import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import login

# Number of workers
num_proc = 8
num_proc_load_dataset = num_proc

# Get the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

def hf_login():
    """Handle Hugging Face login"""
    # Check if token is already set in environment
    if os.environ.get('HF_TOKEN'):
        print("Using HF_TOKEN from environment variables")
        login(os.environ.get('HF_TOKEN'))
        return True
    
    # Otherwise prompt for token
    try:
        token = input("Enter your Hugging Face token (or press Enter to use browser login): ")
        if token:
            login(token=token)
        else:
            login()
        print("Successfully logged in to Hugging Face")
        return True
    except Exception as e:
        print(f"Error logging in: {e}")
        print("You can get your token from https://huggingface.co/settings/tokens")
        return False

if __name__ == '__main__':
    print("Logging in to Hugging Face...")
    if not hf_login():
        print("Failed to log in. Exiting.")
        exit()

    # Load directly from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("nampdn-ai/tiny-codes")
    
    # Print info about the dataset
    print(f"Dataset structure: {dataset}")
    
    # Print a few examples to check structure
    print("\n=== Example samples before processing ===")
    for i in range(3):
        example = dataset['train'][i]
        print(f"\nExample {i+1}:")
        for key, value in example.items():
            # Truncate long values for display
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}... (truncated)")
            else:
                print(f"{key}: {value}")
    
    # Create train/val split
    if 'train' in dataset and 'validation' not in dataset:
        split_dataset = dataset['train'].train_test_split(test_size=0.005, seed=2357, shuffle=True)
        split_dataset['val'] = split_dataset.pop('test')  # rename test to val
    else:
        split_dataset = {
            'train': dataset['train'],
            'val': dataset.get('validation', dataset.get('val', None))
        }
        
        # If there's no validation set, create one
        if split_dataset['val'] is None:
            splits = dataset['train'].train_test_split(test_size=0.005, seed=2357, shuffle=True)
            split_dataset['train'] = splits['train']
            split_dataset['val'] = splits['test']
    
    # Print examples with instruction tags added
    print("\n=== Examples with Instruction Tags ===")
    for i in range(5):
        example = split_dataset['train'][i]
        
        # Check if the dataset has prompt and response columns
        if 'prompt' in example and 'response' in example:
            prompt = example['prompt']
            response = example['response']
            
            # Format with instruction markers
            formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            
            # Print formatted example (truncated if very long)
            print(f"\nFormatted Example {i+1}:")
            if len(formatted_text) > 500:
                print(f"{formatted_text[:500]}... (truncated)")
            else:
                print(formatted_text)
                
            # Also print token count
            token_count = len(enc.encode_ordinary(formatted_text))
            print(f"Token count: {token_count}")
            if token_count > 1020:
                print("⚠️ Warning: This example exceeds the 1024 token context window and will be truncated.")
        else:
            print("Dataset doesn't have 'prompt' and 'response' columns. Available columns:")
            print(list(example.keys()))
            break
    
    # Define processing function
    def process(example):
        # Extract prompt and response based on dataset structure
        if 'prompt' in example and 'response' in example:
            prompt = example['prompt']
            response = example['response']
        else:
            # Adjust this logic based on your dataset structure
            print("Warning: 'prompt' and 'response' columns not found. Check structure and adjust code.")
            return {'ids': [], 'len': 0, 'skip': True}
        
        # Format with instruction markers
        full_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        # Tokenize
        ids = enc.encode_ordinary(full_text)
        
        # Check if length exceeds context window
        if len(ids) > 1020:  # Leave a few tokens for safety
            # Truncate response to fit
            prompt_ids = enc.encode_ordinary(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            response_ids = enc.encode_ordinary(response)
            
            # Calculate how many tokens we can keep from the response
            available_tokens = 1020 - len(prompt_ids)
            if available_tokens <= 0:
                # Prompt alone is too long, skip this sample
                return {'ids': [], 'len': 0, 'skip': True}
            
            # Truncate response to fit
            truncated_response_ids = response_ids[:available_tokens]
            
            # Combine prompt with truncated response
            ids = prompt_ids + truncated_response_ids
        
        ids.append(enc.eot_token)  # Add EOT token
        
        return {'ids': ids, 'len': len(ids), 'skip': False}
    
    # Ask user to proceed
    proceed = input("\nDo you want to proceed with tokenizing and saving the dataset? (y/n): ")
    if proceed.lower() != 'y':
        print("Process aborted.")
        exit()
    
    # Tokenize the dataset
    print("\nTokenizing dataset...")
    tokenized = {}
    for split, dset in split_dataset.items():
        print(f"Processing {split} split...")
        tokenized[split] = dset.map(
            process,
            remove_columns=dset.column_names,  # Remove all original columns
            desc=f"Tokenizing {split} split",
            num_proc=num_proc,
        )
    
    # Filter out skipped samples
    for split in tokenized:
        filtered = tokenized[split].filter(lambda x: not x['skip'])
        tokenized[split] = filtered.remove_columns(['skip'])
        print(f"Filtered {split} dataset: {len(dset)} -> {len(tokenized[split])} samples")
    
    # Save to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        dtype = np.uint16  # Can use uint16 since GPT-2 tokens are < 2^16
        
        print(f"\nWriting {filename}...")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            
            # Skip empty batches
            if len(batch['ids']) == 0:
                continue
                
            arr_batch = np.concatenate(batch['ids'])
            
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        
        arr.flush()
        
        # Print some statistics
        print(f"Split: {split}")
        print(f"Total samples: {len(dset)}")
        print(f"Total tokens: {arr_len}")
        print(f"Average tokens per sample: {arr_len / len(dset)}")