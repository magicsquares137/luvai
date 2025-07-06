from huggingface_hub import list_repo_files, hf_hub_download
import gzip
import json
import math
import random
import zstandard as zstd
import io
import tempfile
import shutil
import psutil
import os
from collections import defaultdict

class GenerateSampleOlmoMix:
    def __init__(self, num_samples: int, seed: int = None) -> None:
        self.num_samples = num_samples

        if seed:
            random.seed(seed)

        self.estimated_characters_per_token = 4
        self.tokens_per_source = {
            "dclm": 52300000000,        # 52.3B
            "starcoder": 1170000000,    # 1.17B
            "pes2o": 825000000,         # 825M  
            "arxiv": 292000000,         # 292M
            "open-web-math": 172000000, # 172M
            "algebraic-stack": 166000000, # 166M
            "wiki": 51000000           # 51M
        }
        self.tokens_per_gz = 150000000 # 150M tokens per file estimate
        self.dataset_name = "allenai/olmo-mix-1124"
        
        # Track stats without storing data
        self.total_examples = 0
        self.total_tokens = 0
        self.source_stats = defaultdict(lambda: {'examples': 0, 'tokens': 0})
        
        # Convert desired tokens to file counts
        self.file_counts = {}
        for key, value in self.tokens_per_source.items():
            self.file_counts[key] = int(math.ceil(value / self.tokens_per_gz))
    
    def check_memory_usage(self):
        """Monitor memory usage and warn if getting high"""
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        if memory_gb > 8:  # Warn if using more than 8GB
            print(f"WARNING: High memory usage: {memory_gb:.1f}GB")
            return False
        return True
    
    def write_example(self, example, output_file):
        """Write example immediately to output file"""
        output_file.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Update stats
        self.total_examples += 1
        tokens = len(example['text']) // self.estimated_characters_per_token
        self.total_tokens += tokens
        self.source_stats[example['src']]['examples'] += 1
        self.source_stats[example['src']]['tokens'] += tokens
        
        # Periodic memory check and progress update
        if self.total_examples % 10000 == 0:
            self.check_memory_usage()
            print(f"    Progress: {self.total_examples:,} examples, {self.total_tokens:,} tokens")
    
    def download_sample(self, output_path: str) -> None:
        print(f"Loading repo files")
        self.files = list_repo_files(self.dataset_name, repo_type="dataset")
        
        sample_id = 0
        
        # Open output file once and keep it open
        with open(output_path, 'w', encoding='utf-8') as output_file:
            
            # Process each source
            for source_key, target_tokens in self.tokens_per_source.items():
                print(f"\n=== Processing {source_key} (target: {target_tokens:,} tokens) ===")
                
                # Get filenames with source_key in them
                if source_key == "dclm":
                    file_subset = [file for file in self.files if source_key in file and file.endswith('.jsonl.zstd')]
                else:
                    file_subset = [file for file in self.files if source_key in file and file.endswith('.json.gz')]
                
                if len(file_subset) == 0:
                    print(f"Unable to find files for: {source_key}")
                    continue
                
                # Calculate how many files we need
                files_needed = self.file_counts[source_key]
                selected_files = random.sample(file_subset, min(files_needed, len(file_subset)))
                
                # Calculate target tokens per file
                tokens_per_file = target_tokens // len(selected_files)
                
                source_tokens_collected = 0
                
                # Process each file
                for file_idx, filename in enumerate(selected_files):
                    print(f"  Processing file {file_idx+1}/{len(selected_files)}: {filename}")
                    current_tokens = 0
                    
                    try:
                        temp_dir = tempfile.mkdtemp()
                        
                        try:
                            file_path = hf_hub_download(
                                repo_id=self.dataset_name, 
                                filename=filename, 
                                repo_type="dataset",
                                cache_dir=temp_dir
                            )
                            
                            # Handle different compression formats
                            if filename.endswith('.jsonl.zstd'):
                                with open(file_path, 'rb') as compressed_file:
                                    dctx = zstd.ZstdDecompressor()
                                    with dctx.stream_reader(compressed_file) as reader:
                                        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                                        for line in text_stream:
                                            if source_tokens_collected >= target_tokens:
                                                break
                                            
                                            try:
                                                example = json.loads(line.strip())
                                                text = example.get('text', '')
                                                
                                                if not text:
                                                    continue
                                                
                                                example_tokens = len(text) // self.estimated_characters_per_token
                                                
                                                if source_tokens_collected + example_tokens <= target_tokens:
                                                    megatron_example = {
                                                        "text": text,
                                                        "src": source_key,
                                                        "type": source_key,
                                                        "id": str(sample_id),
                                                        "title": example.get('id', f"{source_key}_{sample_id}")
                                                    }
                                                    
                                                    # Write immediately instead of storing
                                                    self.write_example(megatron_example, output_file)
                                                    current_tokens += example_tokens
                                                    sample_id += 1
                                                    source_tokens_collected += example_tokens
                                                else:
                                                    break
                                                    
                                            except json.JSONDecodeError:
                                                continue
                            
                            else:
                                # Handle .json.gz files
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    for line in f:
                                        if source_tokens_collected >= target_tokens:
                                            break
                                        
                                        try:
                                            example = json.loads(line.strip())
                                            text = example.get('text', '')
                                            
                                            if not text:
                                                continue
                                            
                                            example_tokens = len(text) // self.estimated_characters_per_token
                                            
                                            if source_tokens_collected + example_tokens <= target_tokens:
                                                megatron_example = {
                                                    "text": text,
                                                    "src": source_key,
                                                    "type": source_key,
                                                    "id": str(sample_id),
                                                    "title": example.get('id', f"{source_key}_{sample_id}")
                                                }
                                                
                                                # Write immediately instead of storing
                                                self.write_example(megatron_example, output_file)
                                                current_tokens += example_tokens
                                                sample_id += 1
                                                source_tokens_collected += example_tokens
                                            else:
                                                break
                                                
                                        except json.JSONDecodeError:
                                            continue
                            
                            print(f"    Sampled {current_tokens:,} tokens from this file")
                            print(f"    Source total so far: {source_tokens_collected:,} tokens")
                            
                        finally:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            
                    except Exception as e:
                        print(f"    Error processing {filename}: {e}")
                        continue
                
                print(f"  Completed {source_key}: {source_tokens_collected:,} tokens")
        
        print(f"\nProcessing complete! Data written to: {output_path}")
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics"""
        print(f"\n=== Final Statistics ===")
        print(f"Total examples: {self.total_examples:,}")
        print(f"Total estimated tokens: {self.total_tokens:,}")
        
        print(f"\nPer-source breakdown:")
        for src, stats in self.source_stats.items():
            print(f"  {src}: {stats['examples']:,} examples, {stats['tokens']:,} tokens")
        
        # Show target vs actual
        print(f"\nTarget vs Actual:")
        for src, target in self.tokens_per_source.items():
            actual = self.source_stats[src]['tokens']
            percentage = (actual / target) * 100 if target > 0 else 0
            print(f"  {src}: {actual:,} / {target:,} tokens ({percentage:.1f}%)")

# Usage example:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sampled olmo-mix-1124 dataset in Megatron format.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path to save the sampled data JSONL file"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=55000000000,
        help="Total number of tokens to sample (default: 55B)"
    )
    
    args = parser.parse_args()
    
    sampler = GenerateSampleOlmoMix(num_samples=args.tokens)
    sampler.download_sample(args.output_path)