import argparse
import os
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

def download_by_column(repo, split, save_dir, group_col, num_samples):
    print(f"Connecting to {repo} (streaming mode)...")
    dataset = load_dataset(repo, split=split, streaming=True)
    
    collected_counts = {}
    
    print(f"Scanning for distinct '{group_col}' to download {num_samples} sample(s) each...")
    
    for row in dataset:
        group_val = row.get(group_col, "default")
        
        if group_val not in collected_counts:
            collected_counts[group_val] = 0
            print(f"Found new group: {group_val}")
            
        if collected_counts[group_val] < num_samples:
            save_path = os.path.join(save_dir, f"noise_{group_val}_sample_{collected_counts[group_val]}.parquet")
            df = pd.DataFrame([row])
            df.to_parquet(save_path)
            collected_counts[group_val] += 1
            print(f"  -> Saved sample to {save_path}")
            
        if len(collected_counts) >= 7 and all(v >= num_samples for v in collected_counts.values()):
            print("Successfully collected samples for all 7 perturbation families!")
            break

def download_by_configs(repo, split, save_dir, num_samples):
    """
    So the creator of libero plus decided it would be smart to use unsupported file type of zip, z01, z02, etc. for the dataset
    Therefore this form of downloading is just not possible so this function is practically useless, but I will keep it here for posterity 
    and in case they change the dataset structure in the future which would be smart, even huggingface does not like it.
    """
    configs = get_dataset_config_names(repo)
    print(f"Found {len(configs)} dataset configs (noise types): {configs}")
    
    for config in configs:
        print(f"\nStreaming config: {config}...")
        dataset = load_dataset(repo, name=config, split=split, streaming=True)
        
        for i, row in enumerate(dataset.take(num_samples)):
            save_path = os.path.join(save_dir, f"noise_{config}_sample_{i}.parquet")
            df = pd.DataFrame([row])
            df.to_parquet(save_path)
            print(f"  -> Saved sample to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a subset of trajectories per LIBERO-plus noise type.")
    
    parser.add_argument("--repo", type=str, default="Sylvest/libero_plus_lerobot",
                        help="Hugging Face dataset repo (e.g., Sylvest/libero_plus_lerobot)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples to download per noise type")
    parser.add_argument("--save_dir", type=str, default="./libero_demo_samples",
                        help="Directory to save the downloaded trajectories")
    parser.add_argument("--mode", type=str, choices=["column", "config"], default="column",
                        help="'column' groups by a specific column (LeRobot). 'config' groups by dataset subsets (RLDS).")
    parser.add_argument("--group_col", type=str, default="task_index",
                        help="If using 'column' mode, which column denotes the noise type or task? (e.g., 'task_index', 'task')")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to pull from (usually 'train')")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == "column":
        download_by_column(args.repo, args.split, args.save_dir, args.group_col, args.samples)
    else:
        download_by_configs(args.repo, args.split, args.save_dir, args.samples)
        
    print("\n✅ Download complete!")