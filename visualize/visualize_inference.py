import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import argparse
from sklearn.metrics import confusion_matrix
import os

def get_args():
    parser = argparse.ArgumentParser(description="Visualize inference results for EGM models")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint to visualize (exclude .chkpt extension)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--time_range', type=int, default=1, help='Time range for visualizations in seconds')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    return parser.parse_args()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def plot_signal_comparison(time, gt_signal, pred_signal, masked_positions, sample_idx, output_path):
    """Plot comparison between ground truth and predicted signal with masked regions highlighted"""
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth
    plt.plot(time, gt_signal, color='blue', alpha=0.7, label='Ground Truth')
    
    # Plot predicted signal
    plt.plot(time, pred_signal, color='red', alpha=0.7, label='Predicted')
    
    # Highlight masked regions
    for pos in masked_positions:
        if pos < len(time):
            plt.axvspan(time[pos]-0.001, time[pos]+0.001, color='gray', alpha=0.3)
    
    # Mark masked positions that were predicted
    plt.scatter(time[masked_positions], pred_signal[masked_positions], 
                color='green', s=25, alpha=0.8, label='Masked Positions')
    
    plt.title(f'Signal Comparison - Sample {sample_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}/signal_comparison_{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_heatmap(attention_weights, sample_idx, layer_idx, head_idx, output_path):
    """Plot attention weights as a heatmap for a given sample, layer and attention head"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, cmap="viridis", 
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title(f'Attention Heatmap - Sample {sample_idx}, Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Token Position')
    plt.ylabel('Token Position')
    plt.savefig(f"{output_path}/attention_heatmap_sample{sample_idx}_layer{layer_idx}_head{head_idx}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_global_attention(global_attention_weights, sample_idx, layer_idx, head_idx, output_path):
    """Plot global attention weights for Longformer model"""
    plt.figure(figsize=(12, 6))
    
    # Average across query tokens for visualization
    avg_global_attention = global_attention_weights.mean(axis=0)
    
    # Plot as a bar chart
    plt.bar(range(len(avg_global_attention)), avg_global_attention)
    
    plt.title(f'Global Attention - Sample {sample_idx}, Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Token Position')
    plt.ylabel('Average Global Attention Weight')
    plt.savefig(f"{output_path}/global_attention_sample{sample_idx}_layer{layer_idx}_head{head_idx}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(gt_labels, pred_labels, output_path):
    """Plot confusion matrix for classification results"""
    cm = confusion_matrix(gt_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f"{output_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and return metrics
    TP = cm[1, 1] if cm.shape[0] > 1 else 0
    FP = cm[0, 1] if cm.shape[0] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0 
    TN = cm[0, 0] if cm.shape[0] > 1 else 0
    
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "PPV": ppv,
        "NPV": npv
    }

def visualize_inference_results(checkpoint_path, num_samples=5, time_range=1, output_dir=None):
    """Main function to visualize inference results"""
    # Load the inference results
    results_path = f"{checkpoint_path}/best_np.npy"
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = f"{checkpoint_path}/visualizations"
    ensure_directory_exists(output_dir)
    
    # Load results
    print(f"Loading results from {results_path}")
    results = np.load(results_path, allow_pickle=True).item()
    
    # Create time array based on signal length
    if 'gt_signals' in results and len(results['gt_signals']) > 0:
        signal_length = len(results['gt_signals'][0])
        time = np.linspace(0, time_range, signal_length)
        
        print(f"Found {len(results['gt_signals'])} signal samples")
        num_to_visualize = min(num_samples, len(results['gt_signals']))
        
        # Plot signal comparisons
        for i in range(num_to_visualize):
            if i < len(results['masked_signals']):
                plot_signal_comparison(
                    time, 
                    results['gt_signals'][i], 
                    results['pred_signals'][i], 
                    results['masked_signals'][i], 
                    i, 
                    output_dir
                )
    else:
        print("No signal data found in results")
    
    # Visualize attention mechanisms if available
    if 'attentions' in results and len(results['attentions']) > 0:
        print(f"Found attention data for {len(results['attentions'])} samples")
        for sample_idx, sample_attention in enumerate(results['attentions']):
            if sample_idx >= num_samples:
                break
                
            # Plot attention for each layer and head
            for layer_idx, layer_attention in enumerate(sample_attention[0]):
                if layer_idx % 2 == 0:  # Plot every other layer to reduce output
                    num_heads = layer_attention.shape[0]
                    for head_idx in range(min(2, num_heads)):  # Plot first 2 heads
                        # Sample down for visualization if matrix is too large
                        attn = layer_attention[head_idx]
                        if attn.shape[0] > 100:
                            step = attn.shape[0] // 100
                            attn = attn[::step, ::step]
                            
                        plot_attention_heatmap(
                            attn,
                            sample_idx,
                            layer_idx,
                            head_idx,
                            output_dir
                        )
    
    # Visualize global attention for Longformer if available
    if 'global_attentions' in results and len(results['global_attentions']) > 0:
        print(f"Found global attention data for {len(results['global_attentions'])} samples")
        for sample_idx, sample_global_attention in enumerate(results['global_attentions']):
            if sample_idx >= num_samples:
                break
                
            # Plot global attention for selected layers and heads
            for layer_idx, layer_global_attention in enumerate(sample_global_attention[0]):
                if layer_idx % 4 == 0:  # Plot fewer layers
                    num_heads = layer_global_attention.shape[0]
                    for head_idx in range(min(1, num_heads)):  # Plot just first head
                        plot_global_attention(
                            layer_global_attention[head_idx],
                            sample_idx,
                            layer_idx,
                            head_idx,
                            output_dir
                        )
    
    # Plot classification results
    if 'gt_afib' in results and 'pred_afib' in results:
        print("Plotting classification metrics")
        metrics = plot_confusion_matrix(
            results['gt_afib'],
            results['pred_afib'],
            output_dir
        )
        
        # Save metrics to text file
        with open(f"{output_dir}/classification_metrics.txt", 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
                print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    args = get_args()
    visualize_inference_results(
        f"./runs/checkpoint/{args.checkpoint}",
        num_samples=args.num_samples,
        time_range=args.time_range,
        output_dir=args.output_dir
    )
    print("Visualization completed successfully!") 