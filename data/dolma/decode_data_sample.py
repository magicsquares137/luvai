import os
import numpy as np
import tiktoken
import argparse

def decode_sample(file_path, sample_index=0, sample_length=100):
    """
    Decode and print a sample from a tokenized binary file
    
    Args:
        file_path: Path to the binary token file
        sample_index: Starting index to read from (default: 0)
        sample_length: Number of tokens to read (default: 100)
    """
    # Load the binary file
    print(f"Loading data from {file_path}...")
    
    # Check file size
    file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Load the tokens
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    print(f"Total tokens in file: {len(data):,}")
    
    # Adjust the sample index if needed
    if sample_index >= len(data):
        print(f"Sample index {sample_index} is out of bounds. Using index 0.")
        sample_index = 0
    
    # Adjust the sample length if needed
    if sample_index + sample_length > len(data):
        sample_length = len(data) - sample_index
        print(f"Adjusted sample length to {sample_length} tokens.")
    
    # Get the token sample
    sample_tokens = data[sample_index:sample_index + sample_length]
    
    # Get the GPT-2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Decode the tokens
    decoded_text = enc.decode(sample_tokens)
    
    # Print information
    print("\n" + "=" * 80)
    print(f"SAMPLE FROM INDEX {sample_index} ({sample_length} TOKENS):")
    print("=" * 80)
    print(decoded_text)
    print("=" * 80)
    
    # Look for document boundaries
    eot_token = enc.eot_token
    # Find positions of EOT tokens in the next 1000 tokens
    check_range = min(1000, len(data) - sample_index)
    extended_sample = data[sample_index:sample_index + check_range]
    eot_positions = np.where(extended_sample == eot_token)[0]
    
    if len(eot_positions) > 0:
        print(f"\nFound EOT tokens at offsets: {eot_positions}")
        
        # If there are at least 2 EOT tokens, show the first complete document
        if len(eot_positions) >= 2:
            doc_start = 0 if sample_index == 0 else eot_positions[0] + 1
            doc_end = eot_positions[1]
            
            if doc_start < doc_end:
                doc_tokens = data[sample_index + doc_start:sample_index + doc_end]
                doc_text = enc.decode(doc_tokens)
                
                print("\n" + "=" * 80)
                print(f"COMPLETE DOCUMENT (TOKENS {doc_start}-{doc_end}, {doc_end-doc_start} TOKENS):")
                print("=" * 80)
                print(doc_text)
                print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode samples from tokenized binary files')
    parser.add_argument('file_path', help='Path to the binary token file')
    parser.add_argument('--index', type=int, default=0, 
                        help='Starting index in the file (default: 0)')
    parser.add_argument('--length', type=int, default=100, 
                        help='Number of tokens to decode (default: 100)')
    
    args = parser.parse_args()
    
    decode_sample(args.file_path, args.index, args.length)
