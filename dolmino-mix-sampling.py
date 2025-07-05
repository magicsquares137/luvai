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
from typing import Dict

class GenerateSampleDolminoMix:
    def __init__(self, num_samples: int, tokens_per_source: Dict = None) -> None:
        self.num_samples = num_samples
        self.estimated_characters_per_token = 4
        if not tokens_per_source:
            self.tokens_per_source = {
                "dclm": 52300000000,        # 52.3B
                "flan": 1170000000,         # 1.17B
                "pes2o": 825000000,         # 825M  
                "arxiv": 292000000,         # 292M
                "math": 172000000,          # 172M
                "stackexchange": 166000000, # 166M
                "wiki": 51000000            # 51M
            }
        else:
            self.tokens_per_source = tokens_per_source
        self.tokens_per_gz = 150000000 # 150M tokens per file estimate
        self.dataset_name = "allenai/dolmino-mix-1124"
        
        # Track stats without storing data
        self.total_examples = 0
        self.total_tokens = 0
        self.source_stats = defaultdict(lambda: {'examples': 0, 'tokens': 0})
        
        # Convert desired tokens to file counts
        self.file_counts = {}
        for key, value in self.tokens_per_source.items():
            self.file_counts[key] = int(math.ceil(value / self.tokens_per_gz))
    
    def get_file_subset_for_source(self, source_key):
        """Get appropriate file subset based on source and file format"""
        if source_key == "dclm":
            # DCLM uses .zst files
            return [file for file in self.files if source_key in file and file.endswith('.zst')]
        elif source_key == "math":
            # Math uses .jsonl files
            return [file for file in self.files if source_key in file and file.endswith('.jsonl')]
        else:
            # Others use .json.gz files
            return [file for file in self.files if source_key in file and file.endswith('.json.gz')]
    
    def process_file_by_format(self, file_path, filename, source_key, tokens_per_file, sample_id, output_file):
        """Process file based on its format"""
        current_tokens = 0
        examples_added = 0
        
        try:
            if filename.endswith('.zst'):
                # Handle zstandard compressed files (dclm)
                with open(file_path, 'rb') as compressed_file:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(compressed_file) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                        for line in text_stream:
                            if current_tokens >= tokens_per_file:
                                break
                            
                            examples_added += self.process_line(
                                line, source_key, sample_id + examples_added, 
                                output_file, tokens_per_file, current_tokens
                            )
                            current_tokens = self.source_stats[source_key]['tokens'] - (
                                self.source_stats[source_key]['tokens'] - current_tokens - 
                                (len(line) // self.estimated_characters_per_token if examples_added > 0 else 0)
                            )
                            
            elif filename.endswith('.jsonl'):
                # Handle plain JSONL files (math)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if current_tokens >= tokens_per_file:
                            break
                        
                        if self.process_line(line, source_key, sample_id + examples_added, output_file, tokens_per_file, current_tokens):
                            examples_added += 1
                            current_tokens += len(line) // self.estimated_characters_per_token
                            
            elif filename.endswith('.json.gz'):
                # Handle gzip compressed JSON files (flan, pes2o, arxiv, stackexchange, wiki)
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        if current_tokens >= tokens_per_file:
                            break
                        
                        if self.process_line(line, source_key, sample_id + examples_added, output_file, tokens_per_file, current_tokens):
                            examples_added += 1
                            current_tokens += len(line) // self.estimated_characters_per_token
            
            return current_tokens, examples_added
            
        except Exception as e:
            print(f"    Error processing file content: {e}")
            return current_tokens, examples_added
    
    def process_line(self, line, source_key, sample_id, output_file, tokens_per_file, current_tokens):
        """Process a single line from any file format"""
        try:
            example = json.loads(line.strip())
            text = example.get('text', '')
            
            if not text:
                return False
            
            example_tokens = len(text) // self.estimated_characters_per_token
            
            if current_tokens + example_tokens <= tokens_per_file:
                megatron_example = {
                    "text": text,
                    "src": source_key,
                    "type": source_key,
                    "id": str(sample_id),
                    "title": example.get('id', f"{source_key}_{sample_id}")
                }
                
                # Write immediately instead of storing
                self.write_example(megatron_example, output_file)
                return True
            else:
                return False
                
        except json.JSONDecodeError:
            return False
    
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
                
                # Get filenames with appropriate extension for this source
                file_subset = self.get_file_subset_for_source(source_key)
                
                if len(file_subset) == 0:
                    print(f"Unable to find files for: {source_key}")
                    print(f"  Available file extensions in repo:")
                    extensions = set(file.split('.')[-1] for file in self.files if source_key in file)
                    print(f"  {extensions}")
                    continue
                
                print(f"  Found {len(file_subset)} files for {source_key}")
                
                # Calculate how many files we need
                files_needed = self.file_counts[source_key]
                selected_files = random.sample(file_subset, min(files_needed, len(file_subset)))
                
                # Calculate target tokens per file
                tokens_per_file = target_tokens // len(selected_files)
                
                source_tokens_collected = 0
                
                # Process each file
                for file_idx, filename in enumerate(selected_files):
                    print(f"  Processing file {file_idx+1}/{len(selected_files)}: {filename}")
                    
                    try:
                        temp_dir = tempfile.mkdtemp()
                        
                        try:
                            file_path = hf_hub_download(
                                repo_id=self.dataset_name, 
                                filename=filename, 
                                repo_type="dataset",
                                cache_dir=temp_dir
                            )
                            
                            # Process file based on its format
                            current_tokens, examples_added = self.process_file_by_format(
                                file_path, filename, source_key, tokens_per_file, sample_id, output_file
                            )
                            
                            sample_id += examples_added
                            source_tokens_collected += current_tokens
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
    
    parser = argparse.ArgumentParser(description="Generate sampled dolmino-mix-1124 dataset in Megatron format.")
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
    
    sampler = GenerateSampleDolminoMix(num_samples=args.tokens)
    sampler.download_sample(args.output_path)