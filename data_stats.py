import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from data_loader import EGMTSDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_args():
    parser = argparse.ArgumentParser(description='Data Statistics')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--signal_size', type=int, default=250, help='Signal size')
    parser.add_argument('--mask', type=float, default=0.75, help='Mask percentage')
    parser.add_argument('--TS', default=False, action=argparse.BooleanOptionalAction, help='Token Substitution')
    parser.add_argument('--TA', default=False, action=argparse.BooleanOptionalAction, help='Token Addition')
    parser.add_argument('--LF', default=False, action=argparse.BooleanOptionalAction, help='Label Flipping')
    parser.add_argument('--output_dir', type=str, default='./data_stats', help='Output directory for plots')
    return parser.parse_args()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")

def analyze_dataset(dataset, name, args):
    print(f"\n===== {name} Dataset Analysis =====")
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize statistics
    afib_labels = []
    signal_mins = []
    signal_maxs = []
    signal_means = []
    signal_stds = []
    nan_count = 0
    
    # Sample indices for signal visualization
    num_samples_to_plot = min(5, len(dataset))
    sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False)
    
    # Create a figure for sample signals
    fig, axes = plt.subplots(num_samples_to_plot, 1, figsize=(10, 3*num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes]
    
    # Iterate through dataset
    for i in tqdm(range(len(dataset)), desc=f"Analyzing {name} dataset"):
        # Get sample
        masked_signal, signal, afib_label, mask, attention_mask = dataset[i]
        
        # Check for NaNs
        if torch.isnan(masked_signal).any() or torch.isnan(signal).any():
            nan_count += 1
        
        # Collect statistics
        afib_labels.append(afib_label)
        signal_mins.append(signal.min().item())
        signal_maxs.append(signal.max().item())
        signal_means.append(signal.mean().item())
        signal_stds.append(signal.std().item())
        
        # Plot sample signals
        if i in sample_indices:
            idx = np.where(i == sample_indices)[0][0]
            axes[idx].plot(signal.numpy())
            axes[idx].set_title(f"Sample {i}, AFib Label: {afib_label}")
            axes[idx].set_ylabel("Amplitude")
            axes[idx].grid(True)
    
    # Add common x-label
    fig.text(0.5, 0.04, 'Time (samples)', ha='center')
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{name}_samples.png")
    plt.close()
    
    # Convert to numpy arrays for analysis
    afib_labels = np.array(afib_labels)
    signal_mins = np.array(signal_mins)
    signal_maxs = np.array(signal_maxs)
    signal_means = np.array(signal_means)
    signal_stds = np.array(signal_stds)
    
    # Print statistics
    print(f"AFib Label Distribution:")
    unique_labels, counts = np.unique(afib_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} ({count/len(afib_labels)*100:.2f}%)")
    
    print(f"Signal Statistics:")
    print(f"  Min: {signal_mins.min():.4f} to {signal_mins.max():.4f}, Mean: {signal_mins.mean():.4f}")
    print(f"  Max: {signal_maxs.min():.4f} to {signal_maxs.max():.4f}, Mean: {signal_maxs.mean():.4f}")
    print(f"  Mean: {signal_means.min():.4f} to {signal_means.max():.4f}, Mean of means: {signal_means.mean():.4f}")
    print(f"  Std: {signal_stds.min():.4f} to {signal_stds.max():.4f}, Mean of stds: {signal_stds.mean():.4f}")
    print(f"  NaN count: {nan_count}/{len(dataset)} ({nan_count/len(dataset)*100:.2f}%)")
    
    # Create histogram of afib labels
    plt.figure(figsize=(8, 6))
    plt.bar(unique_labels, counts)
    plt.title(f"{name} Dataset: AFib Label Distribution")
    plt.xlabel("AFib Label")
    plt.ylabel("Count")
    plt.xticks(unique_labels)
    plt.grid(True, axis='y')
    plt.savefig(f"{args.output_dir}/{name}_afib_distribution.png")
    plt.close()
    
    # Create histograms of signal statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(signal_mins, bins=30)
    axes[0, 0].set_title(f"{name} Dataset: Signal Minimums")
    axes[0, 0].set_xlabel("Min Value")
    axes[0, 0].grid(True)
    
    axes[0, 1].hist(signal_maxs, bins=30)
    axes[0, 1].set_title(f"{name} Dataset: Signal Maximums")
    axes[0, 1].set_xlabel("Max Value")
    axes[0, 1].grid(True)
    
    axes[1, 0].hist(signal_means, bins=30)
    axes[1, 0].set_title(f"{name} Dataset: Signal Means")
    axes[1, 0].set_xlabel("Mean Value")
    axes[1, 0].grid(True)
    
    axes[1, 1].hist(signal_stds, bins=30)
    axes[1, 1].set_title(f"{name} Dataset: Signal Standard Deviations")
    axes[1, 1].set_xlabel("Standard Deviation")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/{name}_signal_stats.png")
    plt.close()
    
    # Print key information
    keys = dataset.keys[:10]  # Print first 10 keys
    print(f"Sample Keys (first 10):")
    for key in keys:
        print(f"  {key}")
    
    return {
        'afib_labels': afib_labels,
        'signal_mins': signal_mins,
        'signal_maxs': signal_maxs,
        'signal_means': signal_means,
        'signal_stds': signal_stds,
        'nan_count': nan_count,
        'keys': dataset.keys
    }

def inspect_keys(dataset, name):
    print(f"\n===== {name} Dataset Key Analysis =====")
    
    # Extract components of keys
    indices = [key[0] for key in dataset.keys]
    placements = [key[1] for key in dataset.keys]
    segments = [key[2] for key in dataset.keys]
    afib_labels = [key[3] for key in dataset.keys]
    
    # Convert to numpy arrays
    indices = np.array(indices)
    placements = np.array(placements)
    segments = np.array(segments)
    afib_labels = np.array(afib_labels)
    
    # Print statistics about key components
    print(f"Key Component Statistics:")
    
    print(f"  Electrode Indices: Unique values {np.unique(indices)}")
    print(f"  Placement Numbers: Min {placements.min()}, Max {placements.max()}, Unique values {len(np.unique(placements))}")
    print(f"  Segment Indices: Min {segments.min()}, Max {segments.max()}, Unique values {len(np.unique(segments))}")
    
    print(f"  AFib Labels in Keys:")
    unique_labels, counts = np.unique(afib_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"    Label {label}: {count} ({count/len(afib_labels)*100:.2f}%)")
    
    # Check if all placements have the same afib label
    by_placement = {}
    for p, a in zip(placements, afib_labels):
        if p not in by_placement:
            by_placement[p] = []
        by_placement[p].append(a)
    
    print("\nAFib Labels by Placement:")
    for p, labels in by_placement.items():
        unique = np.unique(labels)
        if len(unique) == 1:
            print(f"  Placement {p}: All {len(labels)} segments have label {unique[0]}")
        else:
            counts = [np.sum(np.array(labels) == u) for u in unique]
            print(f"  Placement {p}: {len(labels)} segments with mixed labels: {list(zip(unique, counts))}")
    
    return {
        'indices': indices,
        'placements': placements,
        'segments': segments,
        'afib_labels': afib_labels,
        'by_placement': by_placement
    }

def main():
    args = get_args()
    ensure_directory_exists(args.output_dir)
    
    print("Loading datasets...")
    train = np.load('../data/train_intra.npy', allow_pickle=True).item()
    val = np.load('../data/val_intra.npy', allow_pickle=True).item()
    test = np.load('../data/test_intra.npy', allow_pickle=True).item()
    
    print("Creating datasets...")
    train_dataset = EGMTSDataset(train, args=args)
    val_dataset = EGMTSDataset(val, args=args)
    test_dataset = EGMTSDataset(test, args=args)
    
    # Analyze raw keys from the datasets
    print("\n===== Raw Dataset Key Analysis =====")
    train_keys = inspect_keys(train_dataset, "Train")
    val_keys = inspect_keys(val_dataset, "Validation")
    test_keys = inspect_keys(test_dataset, "Test")
    
    # Analyze loaded datasets
    train_stats = analyze_dataset(train_dataset, "Train", args)
    val_stats = analyze_dataset(val_dataset, "Validation", args)
    test_stats = analyze_dataset(test_dataset, "Test", args)
    
    # Compare keys afib labels with dataset afib labels
    for name, dataset_stats, key_stats in [
        ("Train", train_stats, train_keys),
        ("Validation", val_stats, val_keys),
        ("Test", test_stats, test_keys)
    ]:
        print(f"\n===== {name} Dataset Label Comparison =====")
        extracted_afib = dataset_stats['afib_labels']
        key_afib = key_stats['afib_labels']
        
        # Check if there are any differences
        matched = sum(a == b for a, b in zip(extracted_afib, key_afib))
        print(f"Label match: {matched}/{len(extracted_afib)} ({matched/len(extracted_afib)*100:.2f}%)")
        
        if matched != len(extracted_afib):
            print("Mismatched examples (showing first 10):")
            count = 0
            for i, (a, b) in enumerate(zip(extracted_afib, key_afib)):
                if a != b and count < 10:
                    print(f"  Index {i}: Key label {b}, Dataset label {a}")
                    count += 1

if __name__ == "__main__":
    main() 