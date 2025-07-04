import os
import json
import gzip
from tqdm import tqdm
import numpy as np
import tiktoken
from glob import glob

# Get the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

def tokenize_dolma_files(data_dir="/workspace/dolma_data", output_dir="/workspace"):
    """
    Tokenize Dolma dataset files and save as binary files.
    
    Args:
        data_dir: Directory containing Dolma JSON.GZ files
        output_dir: Directory to save the binary token files
    """
    # Find all files matching the pattern
    file_pattern = 'v1_5r2_sample-*.json.gz'
    dolma_files = sorted(glob(os.path.join(data_dir, file_pattern)))
    
    if not dolma_files:
        print(f"No files found matching pattern {file_pattern} in {data_dir}")
        return
    
    print(f"Found {len(dolma_files)} files to process")
    
    # Create train/val split
    val_files = dolma_files[:max(1, int(len(dolma_files) * 0.0005))]
    train_files = [f for f in dolma_files if f not in val_files]
    
    print(f"Using {len(train_files)} files for training and {len(val_files)} files for validation")
    
    # Create output files
    train_file = os.path.join(output_dir, "dolma_train.bin")
    val_file = os.path.join(output_dir, "dolma_val.bin")
    
    # First let's get a sample of the data to understand its structure
    print("Checking data structure...")
    sample_doc = None
    with gzip.open(dolma_files[0], 'rt', encoding='utf-8') as f:
        for line in f:
            sample_doc = json.loads(line)
            break
    
    print("Sample document structure:", sample_doc.keys())
    
    # Estimate token count from a small sample to avoid full scan
    print("Estimating token count from samples...")
    sample_files = dolma_files[:min(5, len(dolma_files))]
    total_docs = 0
    total_tokens = 0
    
    for file in tqdm(sample_files):
        docs_in_file = 0
        tokens_in_file = 0
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                docs_in_file += 1
                try:
                    doc = json.loads(line)
                    if 'text' in doc and doc['text']:
                        ids = enc.encode_ordinary(doc['text'])
                        tokens_in_file += len(ids) + 1  # +1 for EOT token
                except Exception as e:
                    print(f"Error in file {file}: {e}")
                    continue
        
        total_docs += docs_in_file
        total_tokens += tokens_in_file
    
    # Estimate total tokens
    avg_tokens_per_file = total_tokens / len(sample_files)
    estimated_train_tokens = int(avg_tokens_per_file * len(train_files) * 1.1)  # Add 10% buffer
    estimated_val_tokens = int(avg_tokens_per_file * len(val_files) * 1.1)  # Add 10% buffer
    
    print(f"Average docs per file: {total_docs / len(sample_files):.2f}")
    print(f"Average tokens per file: {avg_tokens_per_file:.2f}")
    print(f"Estimated training tokens: {estimated_train_tokens}")
    print(f"Estimated validation tokens: {estimated_val_tokens}")
    
    # Create memory-mapped arrays
    print(f"Creating training array with {estimated_train_tokens} tokens")
    train_arr = np.memmap(train_file, dtype=np.uint16, mode='w+', shape=(estimated_train_tokens,))
    
    print(f"Creating validation array with {estimated_val_tokens} tokens")
    val_arr = np.memmap(val_file, dtype=np.uint16, mode='w+', shape=(estimated_val_tokens,))
    
    # Process training files
    train_idx = 0
    print("Processing training files...")
    for file in tqdm(train_files):
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if 'text' in doc and doc['text']:
                        ids = enc.encode_ordinary(doc['text'])
                        ids.append(enc.eot_token)
                        
                        # Check if we need to resize
                        if train_idx + len(ids) > estimated_train_tokens:
                            new_size = int(estimated_train_tokens * 1.5)
                            print(f"Resizing training array to {new_size} tokens")
                            train_arr.flush()
                            new_train_arr = np.memmap(train_file, dtype=np.uint16, mode='r+', shape=(new_size,))
                            new_train_arr[:train_idx] = train_arr[:train_idx]
                            train_arr = new_train_arr
                            estimated_train_tokens = new_size
                        
                        train_arr[train_idx:train_idx+len(ids)] = ids
                        train_idx += len(ids)
                except Exception as e:
                    print(f"Error processing line in {file}: {e}")
                    continue
    
    # Process validation files
    val_idx = 0
    print("Processing validation files...")
    for file in tqdm(val_files):
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if 'text' in doc and doc['text']:
                        ids = enc.encode_ordinary(doc['text'])
                        ids.append(enc.eot_token)
                        
                        # Check if we need to resize
                        if val_idx + len(ids) > estimated_val_tokens:
                            new_size = int(estimated_val_tokens * 1.5)
                            print(f"Resizing validation array to {new_size} tokens")
                            val_arr.flush()
                            new_val_arr = np.memmap(val_file, dtype=np.uint16, mode='r+', shape=(new_size,))
                            new_val_arr[:val_idx] = val_arr[:val_idx]
                            val_arr = new_val_arr
                            estimated_val_tokens = new_size
                        
                        val_arr[val_idx:val_idx+len(ids)] = ids
                        val_idx += len(ids)
                except Exception as e:
                    print(f"Error processing line in {file}: {e}")
                    continue
    
    # Trim arrays to actual size
    print("Trimming arrays to final size...")
    train_arr.flush()
    final_train_arr = np.memmap(train_file, dtype=np.uint16, mode='r+', shape=(train_idx,))
    final_train_arr[:] = train_arr[:train_idx]
    final_train_arr.flush()
    
    val_arr.flush()
    final_val_arr = np.memmap(val_file, dtype=np.uint16, mode='r+', shape=(val_idx,))
    final_val_arr[:] = val_arr[:val_idx]
    final_val_arr.flush()
    
    print(f"Tokenization complete! Files saved to {train_file} and {val_file}")
    print(f"Final training tokens: {train_idx}, Final validation tokens: {val_idx}")
    
    # Calculate size in GB
    train_size_gb = os.path.getsize(train_file) / (1024 ** 3)
    val_size_gb = os.path.getsize(val_file) / (1024 ** 3)
    print(f"Training file size: {train_size_gb:.2f} GB")
    print(f"Validation file size: {val_size_gb:.2f} GB")

if __name__ == "__main__":
    tokenize_dolma_files()