import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers
num_proc = 8
num_proc_load_dataset = num_proc

# Get the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the No Robots dataset
    print("Loading No Robots dataset...")
    dataset = load_dataset("HuggingFaceH4/no_robots")
    
    print(f"Dataset structure: {dataset}")
    
    # Print a few examples to check structure
    print("\n=== Example samples before processing ===")
    for i in range(3):
        example = dataset['train'][i]
        print(f"\nExample {i+1}:")
        for key, value in example.items():
            if key == 'messages':
                print(f"Messages:")
                for msg in value:
                    # Truncate long content for display
                    content = msg['content']
                    if len(content) > 100:
                        print(f"  - Role: {msg['role']}, Content: {content[:100]}... (truncated)")
                    else:
                        print(f"  - Role: {msg['role']}, Content: {content}")
            else:
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    print(f"{key}: {value[:100]}... (truncated)")
                else:
                    print(f"{key}: {value}")
    
    # Create dataset splits
    split_dataset = {
        'train': dataset['train'],
        'val': dataset['test']  # Use test split as validation
    }
    
    # Print examples with instruction tags added
    print("\n=== Examples with Instruction Tags ===")
    for i in range(5):
        example = split_dataset['train'][i]
        
        # Format with instruction markers
        formatted_text = ""
        for msg in example['messages']:
            role = msg['role']
            content = msg['content']
            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Print formatted example (truncated if very long)
        print(f"\nFormatted Example {i+1}:")
        if len(formatted_text) > 500:
            print(f"{formatted_text[:500]}... (truncated)")
        else:
            print(formatted_text)
        
        # Also print token count
        token_count = len(enc.encode(formatted_text, allowed_special={"<|im_start|>", "<|im_end|>"}))
        print(f"Token count: {token_count}")
        if token_count > 1020:
            print("⚠️ Warning: This example exceeds the 1024 token context window and will be skipped.")
    
    # Define processing function
    def process(example):
        # Format with instruction markers
        full_text = ""
        for msg in example['messages']:
            role = msg['role']
            content = msg['content']
            full_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Tokenize
        ids = enc.encode(full_text, allowed_special={"<|im_start|>", "<|im_end|>"})
        
        # Check if length exceeds context window
        if len(ids) > 1020:  # Leave a few tokens for safety
            # Skip examples that exceed context window
            return {'ids': [], 'len': 0, 'skip': True}
        
        # Add EOT token
        ids.append(enc.eot_token)  # Add EOT token
        
        return {'ids': ids, 'len': len(ids), 'skip': False}
    
    # Ask user to proceed
    proceed = input("\nDo you want to proceed with tokenizing and saving the dataset? (y/n): ")
    if proceed.lower() != 'y':
        print("Process aborted.")
        exit()
    
    # Create output directory
    output_dir = 'data/no_robots'
    os.makedirs(output_dir, exist_ok=True)
    
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
        print(f"Filtered {split} dataset: {len(split_dataset[split])} -> {len(tokenized[split])} samples")
    
    # Save to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
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