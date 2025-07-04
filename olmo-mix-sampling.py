from huggingface_hub import list_repo_files, hf_hub_download
import gzip
import json
import math
import random

class GenerateSampleOlmoMix:
    def __init__(self, num_samples: int) -> None:
        self.num_samples = num_samples
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
        self.tokens_per_gz = 150000000
        self.dataset_name = "allenai/olmo-mix-1124"
        self.sampled_data = []
        
        # Convert desired tokens to file counts
        self.file_counts = {}
        for key, value in self.tokens_per_source.items():
            self.file_counts[key] = int(math.ceil(value / self.tokens_per_gz))
    
    def download_sample(self) -> None:
        print(f"Loading repo files")
        self.files = list_repo_files(self.dataset_name, repo_type="dataset")
        
        sample_id = 0
        
        # Process each source
        for source_key, target_tokens in self.tokens_per_source.items():
            print(f"\n=== Processing {source_key} (target: {target_tokens:,} tokens) ===")
            
            # Get filenames with source_key in them
            if source_key == "dclm":
                # DCLM files use .jsonl.zstd format
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
            
            # Process each file
            for filename in selected_files:
                print(f"  Processing: {filename}")
                
                try:
                    # Use /tmp for downloads, process immediately, then delete
                    import tempfile
                    import shutil

                    temp_dir = tempfile.mkdtemp()

                    try:
                        file_path = hf_hub_download(
                            repo_id=self.dataset_name, 
                            filename=filename, 
                            repo_type="dataset",
                            cache_dir=temp_dir
                        )

                        current_tokens = 0

                        # Handle different compression formats
                        if filename.endswith('.jsonl.zstd'):
                            import zstandard as zstd
                            with open(file_path, 'rb') as compressed_file:
                                dctx = zstd.ZstdDecompressor()
                                with dctx.stream_reader(compressed_file) as reader:
                                    text_reader = reader.read().decode('utf-8')
                                    for line in text_reader.splitlines():
                                        if current_tokens >= tokens_per_file:
                                            break

                                        try:
                                            example = json.loads(line.strip())
                                            text = example.get('text', '')

                                            if not text:
                                                continue

                                            example_tokens = len(text) // self.estimated_characters_per_token

                                            if current_tokens + example_tokens <= tokens_per_file:
                                                megatron_example = {
                                                    "text": text,
                                                    "src": source_key,
                                                    "type": source_key,
                                                    "id": str(sample_id),
                                                    "title": example.get('id', f"{source_key}_{sample_id}")
                                                }

                                                self.sampled_data.append(megatron_example)
                                                current_tokens += example_tokens
                                                sample_id += 1
                                            else:
                                                break

                                        except json.JSONDecodeError:
                                            continue

                        else:
                            # Handle .json.gz files
                            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                for line in f:
                                    if current_tokens >= tokens_per_file:
                                        break

                                    try:
                                        example = json.loads(line.strip())
                                        text = example.get('text', '')

                                        if not text:
                                            continue

                                        example_tokens = len(text) // self.estimated_characters_per_token

                                        if current_tokens + example_tokens <= tokens_per_file:
                                            megatron_example = {
                                                "text": text,
                                                "src": source_key,
                                                "type": source_key,
                                                "id": str(sample_id),
                                                "title": example.get('id', f"{source_key}_{sample_id}")
                                            }

                                            self.sampled_data.append(megatron_example)
                                            current_tokens += example_tokens
                                            sample_id += 1
                                        else:
                                            break

                                    except json.JSONDecodeError:
                                        continue

                            print(f"    Sampled {current_tokens:,} tokens from this file")

                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)

                except Exception as e:
                    print(f"    Error processing {filename}: {e}")
                    continue

    
    def save_megatron_format(self, output_path: str):
        """Save in loose JSON format for Megatron preprocessing"""
        print(f"\nSaving {len(self.sampled_data)} examples to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.sampled_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved successfully!")
        
        # Print some stats
        total_chars = sum(len(ex['text']) for ex in self.sampled_data)
        estimated_total_tokens = total_chars // self.estimated_characters_per_token
        
        print(f"Total examples: {len(self.sampled_data):,}")
        print(f"Estimated total tokens: {estimated_total_tokens:,}")
        
        # Per-source stats
        source_counts = {}
        for ex in self.sampled_data:
            src = ex['src']
            if src not in source_counts:
                source_counts[src] = 0
            source_counts[src] += len(ex['text']) // self.estimated_characters_per_token
        
        print("\nPer-source token counts:")
        for src, tokens in source_counts.items():
            print(f"  {src}: {tokens:,} tokens")

# Usage example:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sampled olmo-mix-1124 dataset in Megatron format.")
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path to save the sampled data JSONL file"
    )

    args = parser.parse_args()

    sampler = GenerateSampleOlmoMix(num_samples=55000000000)  # 55B tokens
    sampler.download_sample()
    sampler.save_megatron_format(args.output_path)