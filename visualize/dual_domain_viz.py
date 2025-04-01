import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch
import argparse
import os
import sys

# Adjust the path for imports from the main directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)  # Add the project directory to path

from frequency_tokenizer import FrequencyTokenizer
from dual_path_model import DualPathECGModel
from transformers import BigBirdConfig, LongformerConfig, BigBirdTokenizer, LongformerTokenizer
from captum.attr import LayerIntegratedGradients, IntegratedGradients


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--model_file', type=str, default='best_model.pt', help='Model file name')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    parser.add_argument('--model', type=str, default='big', choices=['big', 'long'], help='Model type')
    parser.add_argument('--signal_size', type=int, default=250, help='Signal size for quantization')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--TS', action='store_true', help='Enable Token Substitution')
    parser.add_argument('--TA', action='store_true', help='Enable Token Addition')
    parser.add_argument('--LF', action='store_true', help='Enable Label Flipping')
    parser.add_argument('--CF', action='store_true', help='Implement Counterfactuals')
    parser.add_argument('--key', type=str, default=None, help='Specific key to visualize')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    return parser.parse_args()


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def moving_average(signal, window_size=50):
    """Apply moving average to signal"""
    return np.convolve(signal, np.ones(window_size), 'same') / window_size


def label_flip(afib_label):
    """Flip AFib label"""
    if afib_label == 0:
        afib_label = 1
    elif afib_label == 1:
        afib_label = 0
    return afib_label


def sample_consecutive(signal, sample_size):
    """Sample consecutive values from signal"""
    max_start_index = len(signal) - sample_size
    start_index = np.random.randint(0, max_start_index)
    return signal[start_index:start_index + sample_size]


def preprocess_signal(signal, args, afib_label, tokenizer, freq_tokenizer):
    """Preprocess signal into time and frequency domain tokens"""
    # Apply counterfactual transformations if enabled
    if args.TS and args.CF:
        signal = moving_average(signal)
    if args.LF and args.CF:
        afib_label = label_flip(afib_label)
    
    # Get AFib token
    afib_token = f"afib_{int(afib_label)}"
    
    # Process time domain
    min_val, max_val = np.min(signal), np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    quantized_signal = np.floor(normalized_signal * args.signal_size).astype(int)
    quantized_signal_tokens = [f"signal_{i}" for i in quantized_signal]
    quantized_signal_ids = tokenizer.convert_tokens_to_ids(quantized_signal_tokens)
    
    # Process frequency domain
    freq_tokens, spec_min, spec_max, quantized_spec, freqs, times = freq_tokenizer.tokenize(signal)
    freq_token_ids = tokenizer.convert_tokens_to_ids(freq_tokens)
    
    # Get token IDs
    cls_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    afib_id = tokenizer.convert_tokens_to_ids([afib_token])
    
    # Handle token addition if enabled
    if args.TA:
        # Time domain TA
        quantized_augsignal_tokens = [f"augsig_{i}" for i in quantized_signal]
        sampled_quantized_augsignal_tokens = sample_consecutive(
            quantized_augsignal_tokens, int(0.25 * len(quantized_signal_ids))
        )
        sampled_quantized_augsignal_ids = tokenizer.convert_tokens_to_ids(
            sampled_quantized_augsignal_tokens
        )
        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        
        # Create time domain tokens
        time_tokens = [cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
        if args.CF:
            time_tokens2 = [cls_id] + quantized_signal_ids + sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]
        else:
            time_tokens2 = [cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
        
        # Frequency domain TA
        freq_tokens_list = [cls_id] + freq_token_ids + [pad_id] * int(0.25 * len(freq_token_ids)) + [sep_id] + afib_id + [sep_id]
        if args.CF:
            sampled_freq_tokens = sample_consecutive(
                freq_tokens, int(0.25 * len(freq_token_ids))
            )
            sampled_freq_token_ids = tokenizer.convert_tokens_to_ids(sampled_freq_tokens)
            freq_tokens_list2 = [cls_id] + freq_token_ids + sampled_freq_token_ids + [sep_id] + afib_id + [sep_id]
        else:
            freq_tokens_list2 = [cls_id] + freq_token_ids + [pad_id] * int(0.25 * len(freq_token_ids)) + [sep_id] + afib_id + [sep_id]
    else:
        time_tokens = [cls_id] + quantized_signal_ids + [sep_id] + afib_id + [sep_id]
        time_tokens2 = time_tokens
        freq_tokens_list = [cls_id] + freq_token_ids + [sep_id] + afib_id + [sep_id]
        freq_tokens_list2 = freq_tokens_list
    
    # Create masks for time domain (75% masking)
    time_mask = np.ones_like(time_tokens)
    mask_indices_time = np.random.choice(
        len(quantized_signal_ids), int(0.75 * len(quantized_signal_ids)), replace=False
    )
    time_mask[1:len(quantized_signal_ids)+1][mask_indices_time] = 0
    time_mask[-2] = 0  # Always mask the AFib label
    
    # Create masks for frequency domain (75% masking)
    freq_mask = np.ones_like(freq_tokens_list)
    mask_indices_freq = np.random.choice(
        len(freq_token_ids), int(0.75 * len(freq_token_ids)), replace=False
    )
    freq_mask[1:len(freq_token_ids)+1][mask_indices_freq] = 0
    freq_mask[-2] = 0  # Always mask the AFib label
    
    # Create attention masks
    time_attention_mask = np.ones_like(time_tokens)
    freq_attention_mask = np.ones_like(freq_tokens_list)
    
    # Apply masks
    if args.TA:
        time_masked_sample = np.copy(time_tokens2)
        freq_masked_sample = np.copy(freq_tokens_list2)
    else:
        time_masked_sample = np.copy(time_tokens)
        freq_masked_sample = np.copy(freq_tokens_list)
        
    time_masked_sample[time_mask == 0] = mask_id
    freq_masked_sample[freq_mask == 0] = mask_id
    
    # Create global attention masks for Longformer if needed
    if args.model == 'long':
        time_global_attention_mask = np.zeros_like(time_tokens)
        time_global_attention_mask[0] = 1  # Global attention on CLS token
        time_global_attention_mask[-2] = 1  # Global attention on AFib token
        
        freq_global_attention_mask = np.zeros_like(freq_tokens_list)
        freq_global_attention_mask[0] = 1  # Global attention on CLS token
        freq_global_attention_mask[-2] = 1  # Global attention on AFib token
    else:
        time_global_attention_mask = None
        freq_global_attention_mask = None
    
    result = {
        'time_input_ids': time_masked_sample,
        'freq_input_ids': freq_masked_sample,
        'time_labels': time_tokens,
        'freq_labels': freq_tokens_list,
        'time_attention_mask': time_attention_mask,
        'freq_attention_mask': freq_attention_mask,
        'time_mask': time_mask,
        'freq_mask': freq_mask,
        'time_global_attention_mask': time_global_attention_mask,
        'freq_global_attention_mask': freq_global_attention_mask,
        'quantized_spec': quantized_spec,
        'freqs': freqs,
        'times': times,
        'min_val': min_val,
        'max_val': max_val,
        'spec_min': spec_min,
        'spec_max': spec_max
    }
    
    return result, afib_label


def get_integrated_gradients(model, device, signal_data, afib_label):
    """Calculate integrated gradients for both time and frequency domains"""
    # Prepare inputs
    time_input_ids = torch.LongTensor(signal_data['time_input_ids']).unsqueeze(0).to(device)
    freq_input_ids = torch.LongTensor(signal_data['freq_input_ids']).unsqueeze(0).to(device)
    time_attention_mask = torch.tensor(signal_data['time_attention_mask'], dtype=torch.int).unsqueeze(0).to(device)
    freq_attention_mask = torch.tensor(signal_data['freq_attention_mask'], dtype=torch.int).unsqueeze(0).to(device)
    
    # Prepare global attention masks if using Longformer
    time_global_attention_mask = signal_data['time_global_attention_mask']
    freq_global_attention_mask = signal_data['freq_global_attention_mask']
    if time_global_attention_mask is not None:
        time_global_attention_mask = torch.tensor(time_global_attention_mask, dtype=torch.int).unsqueeze(0).to(device)
        freq_global_attention_mask = torch.tensor(freq_global_attention_mask, dtype=torch.int).unsqueeze(0).to(device)
    
    # Create baseline inputs (all padding tokens)
    time_baseline = torch.full_like(time_input_ids, model.time_model.config.pad_token_id)
    freq_baseline = torch.full_like(freq_input_ids, model.freq_model.config.pad_token_id)
    
    # Keep CLS and SEP tokens the same as original
    time_baseline[:, 0] = time_input_ids[:, 0]  # CLS token
    time_baseline[:, -1] = time_input_ids[:, -1]  # Last SEP token
    time_baseline[:, -3] = time_input_ids[:, -3]  # First SEP token
    
    freq_baseline[:, 0] = freq_input_ids[:, 0]  # CLS token
    freq_baseline[:, -1] = freq_input_ids[:, -1]  # Last SEP token
    freq_baseline[:, -3] = freq_input_ids[:, -3]  # First SEP token
    
    # Set up integrated gradients for classification task
    def forward_func(time_ids, time_attn, freq_ids, freq_attn, time_global=None, freq_global=None):
        outputs = model(
            time_input_ids=time_ids,
            time_attention_mask=time_attn,
            freq_input_ids=freq_ids,
            freq_attention_mask=freq_attn,
            time_global_attention_mask=time_global,
            freq_global_attention_mask=freq_global,
            class_labels=None
        )
        return outputs['classification_logits']
    
    # Create integrated gradients instance
    ig = IntegratedGradients(forward_func)
    
    # Calculate attributions
    target_class = int(afib_label)
    
    if time_global_attention_mask is not None:
        attributions = ig.attribute(
            inputs=(
                time_input_ids, 
                time_attention_mask,
                freq_input_ids,
                freq_attention_mask,
                time_global_attention_mask,
                freq_global_attention_mask
            ),
            baselines=(
                time_baseline,
                torch.zeros_like(time_attention_mask),
                freq_baseline,
                torch.zeros_like(freq_attention_mask),
                torch.zeros_like(time_global_attention_mask),
                torch.zeros_like(freq_global_attention_mask)
            ),
            target=target_class,
            n_steps=50
        )
    else:
        attributions = ig.attribute(
            inputs=(
                time_input_ids, 
                time_attention_mask,
                freq_input_ids,
                freq_attention_mask
            ),
            baselines=(
                time_baseline,
                torch.zeros_like(time_attention_mask),
                freq_baseline,
                torch.zeros_like(freq_attention_mask)
            ),
            target=target_class,
            n_steps=50
        )
    
    # Process attributions - sum across dimensions
    time_attributions = attributions[0].sum(dim=-1).squeeze(0).detach().cpu().numpy()
    freq_attributions = attributions[2].sum(dim=-1).squeeze(0).detach().cpu().numpy()
    
    # Normalize attributions
    if time_attributions.max() != 0:
        time_attributions = time_attributions / time_attributions.max()
    if freq_attributions.max() != 0:
        freq_attributions = freq_attributions / freq_attributions.max()
    
    # Apply temperature scaling to attributions
    if hasattr(model, 'temperature'):
        temperature = model.temperature.item()
        time_attributions = time_attributions * temperature  # Multiply by temperature to get true attributions
        freq_attributions = freq_attributions * temperature  # Multiply by temperature to get true attributions
    
    return time_attributions, freq_attributions


def visualize_dual_domain(signal, time_attributions, freq_attributions, freqs, times, 
                          quantized_spec, signal_data, key, args, model, cf_label=""):
    """Visualize time and frequency domain attributions side by side"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
    
    # Time domain visualization
    ax1.plot(signal, label='Signal', color='blue', alpha=0.7)
    mask = signal_data['time_mask']
    masked_indices = np.where(mask[1:len(signal)+1] == 0)[0]
    ax1.scatter(masked_indices, [signal[i] for i in masked_indices], 
                color='red', zorder=2, label='Masked Tokens', s=20, alpha=0.8)
    
    ax1_twin = ax1.twinx()
    # Only show attributions for the actual signal part (skip CLS, SEP, etc.)
    signal_time_attributions = time_attributions[1:len(signal)+1]
    ax1_twin.fill_between(range(len(signal)), 0, signal_time_attributions, 
                        color='green', alpha=0.3, label='Time Domain Attribution')
    
    ax1.set_title('Time Domain: Original Signal with Attributions')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1_twin.set_ylabel('Attribution Score')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Frequency domain visualization - spectrogram with attribution overlay
    spec = signal_data['quantized_spec']
    normalized_spec = spec / args.signal_size  # Convert back to 0-1 range
    
    # Plot spectrogram
    im = ax2.pcolormesh(times, freqs, normalized_spec, shading='gouraud', cmap='viridis')
    ax2.set_title('Frequency Domain: Spectrogram')
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [s]')
    fig.colorbar(im, ax=ax2, label='Magnitude')
    
    # Reshape frequency attributions to match spectrogram if possible
    if len(freq_attributions) >= len(freqs) * len(times):
        freq_attr_reshaped = np.zeros((len(freqs), len(times)))
        idx = 1  # Skip CLS token
        for i in range(len(freqs)):
            for j in range(len(times)):
                if idx < len(freq_attributions):
                    freq_attr_reshaped[i, j] = freq_attributions[idx]
                    idx += 1
        
        # Plot reshaped attributions
        im2 = ax3.pcolormesh(times, freqs, freq_attr_reshaped, shading='gouraud', cmap='Reds')
        ax3.set_title('Frequency Domain: Attribution Map')
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_xlabel('Time [s]')
        fig.colorbar(im2, ax=ax3, label='Attribution Score')
    else:
        # Just plot the raw attribution scores if reshaping isn't possible
        ax3.plot(freq_attributions[1:min(len(freq_attributions), 1000)], color='red')
        ax3.set_title('Frequency Domain: Raw Attribution Scores')
        ax3.set_xlabel('Token Index')
        ax3.set_ylabel('Attribution Score')
    
    # Add temperature information
    if hasattr(model, 'temperature'):
        temperature = model.temperature.item()
        ax4.text(0.5, 0.5, f'Model Temperature: {temperature:.4f}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Model Temperature')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Construct filename with counterfactual information
    cf_info = "_CF" if args.CF else ""
    aug_info = f"_TS{args.TS}_TA{args.TA}_LF{args.LF}"
    
    filename = f"dual_domain_{key}_{cf_label}{aug_info}{cf_info}.png"
    plt.savefig(os.path.join(args.output_dir, filename))
    plt.close()


def main():
    # Get command line arguments
    args = get_args()
    
    # Construct proper checkpoint path
    checkpoint_base = os.path.join("./runs/checkpoint/dual_path", args.checkpoint)
    # if not os.path.exists(checkpoint_base):
    #     # Try looking in just the provided path directly
    #     if os.path.exists(args.checkpoint):
    #         checkpoint_base = args.checkpoint
    #     else:
    #         print(f"Warning: Checkpoint directory not found at {checkpoint_base}")
    #         print(f"Trying alternative paths...")
            
    #         # Try alternate paths
    #         alt_path = os.path.join("../runs/checkpoint", args.checkpoint)
    #         if os.path.exists(alt_path):
    #             checkpoint_base = alt_path
    #         else:
    #             print(f"Error: Could not locate checkpoint directory. Tried:")
    #             print(f"  - {checkpoint_base}")
    #             print(f"  - {args.checkpoint}")
    #             print(f"  - {alt_path}")
    #             sys.exit(1)
            
    print(f"Using checkpoint base directory: {checkpoint_base}")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(checkpoint_base, 'visualizations')
    ensure_directory_exists(args.output_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint_path = os.path.join(checkpoint_base, args.model_file)
    
    if not os.path.exists(checkpoint_path):
        # Try looking for best_checkpoint.chkpt if best_model.pt is not found
        alt_checkpoint_path = os.path.join(checkpoint_base, 'best_checkpoint.chkpt')
        if os.path.exists(alt_checkpoint_path):
            checkpoint_path = alt_checkpoint_path
            print(f"Using alternative checkpoint file: {checkpoint_path}")
        else:
            print(f"Error: Model file not found at {checkpoint_path} or {alt_checkpoint_path}")
            print(f"Available files in {checkpoint_base}:")
            for file in os.listdir(checkpoint_base):
                print(f"  - {file}")
            sys.exit(1)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get saved args if available
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        # Override some parameters with saved ones
        args.signal_size = saved_args.signal_size
        args.model = saved_args.model
        print(f"Using model type '{args.model}' and signal size {args.signal_size} from checkpoint")
    
    # Initialize tokenizers
    print("Initializing tokenizers...")
    if args.model == 'big':
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        time_config = BigBirdConfig()
        time_config.attention_type = 'original_full'
        freq_config = BigBirdConfig()
        freq_config.attention_type = 'original_full'
    elif args.model == 'long':
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        time_config = LongformerConfig()
        freq_config = LongformerConfig()
        
        # Set higher max position embeddings to match training configuration
        time_config.max_position_embeddings = 8192
        freq_config.max_position_embeddings = 8192
        
        # Also set attention window size appropriately
        time_config.attention_window = [512] * time_config.num_hidden_layers
        freq_config.attention_window = [512] * freq_config.num_hidden_layers
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    # Create custom tokens
    custom_tokens = [
        f"signal_{i}" for i in range(args.signal_size+1)
    ] + [
        f"afib_{i}" for i in range(2)
    ] + [
        f"freq_{i}_{j}" for i in range(50) for j in range(args.signal_size+1)
    ]
    
    if args.TA:
        custom_tokens += [
            f"augsig_{i}" for i in range(args.signal_size+1)
        ]
    
    # Add tokens to tokenizer
    tokenizer.add_tokens(custom_tokens)
    
    # Update vocab size in configs
    time_config.vocab_size = len(tokenizer)
    freq_config.vocab_size = len(tokenizer)
    
    # Initialize frequency tokenizer
    freq_tokenizer = FrequencyTokenizer(
        n_freq_bins=50,
        signal_size=args.signal_size,
        sampling_rate=1000
    )
    
    # Initialize model
    print("Initializing model...")
    model = DualPathECGModel(
        time_model_config=time_config,
        freq_model_config=freq_config,
        model_type=args.model,
        fusion_dim=768
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test data...")
    # Only load AFib data since that's all we have
    # test = np.load('../data/test_data_by_placement_na_True_True_True_False_True.npy', allow_pickle=True).item()
    test = np.load('../data/test_intra.npy', allow_pickle=True).item()

    
    # Get keys to process
    if args.key is not None:
        # Use specific key
        keys_to_process = [args.key]
    else:
        # Use first N samples
        keys_list = list(test.keys())
        keys_to_process = keys_list[:args.num_samples]
    
    # Process each key
    print(f"Processing {len(keys_to_process)} samples...")
    for key in tqdm(keys_to_process):
        # Get signal data
        signal = test[key][:1000]
        afib_label = key[3]
        
        # First, process the original signal
        signal_data, afib_label = preprocess_signal(signal, args, afib_label, tokenizer, freq_tokenizer)
        
        # Get attributions
        time_attr, freq_attr = get_integrated_gradients(model, device, signal_data, afib_label)
        
        # Visualize original signal
        visualize_dual_domain(
            signal, time_attr, freq_attr, 
            signal_data['freqs'], signal_data['times'], 
            signal_data['quantized_spec'],
            signal_data, key, args, model, "original"
        )
        
        # If counterfactual analysis is enabled, also process counterfactuals
        if args.CF:
            # TS counterfactual - smoothed signal
            if args.TS:
                ts_args = argparse.Namespace(**vars(args))
                ts_args.TS = True
                ts_args.CF = True
                ts_args.TA = False
                ts_args.LF = False
                
                ts_signal_data, ts_afib_label = preprocess_signal(signal, ts_args, afib_label, tokenizer, freq_tokenizer)
                ts_time_attr, ts_freq_attr = get_integrated_gradients(model, device, ts_signal_data, ts_afib_label)
                
                visualize_dual_domain(
                    moving_average(signal), ts_time_attr, ts_freq_attr, 
                    ts_signal_data['freqs'], ts_signal_data['times'], 
                    ts_signal_data['quantized_spec'],
                    ts_signal_data, key, ts_args, model, "TS"
                )
            
            # TA counterfactual - added tokens
            if args.TA:
                ta_args = argparse.Namespace(**vars(args))
                ta_args.TA = True
                ta_args.CF = True
                ta_args.TS = False
                ta_args.LF = False
                
                ta_signal_data, ta_afib_label = preprocess_signal(signal, ta_args, afib_label, tokenizer, freq_tokenizer)
                ta_time_attr, ta_freq_attr = get_integrated_gradients(model, device, ta_signal_data, ta_afib_label)
                
                visualize_dual_domain(
                    signal, ta_time_attr, ta_freq_attr, 
                    ta_signal_data['freqs'], ta_signal_data['times'], 
                    ta_signal_data['quantized_spec'],
                    ta_signal_data, key, ta_args, model, "TA"
                )
            
            # LF counterfactual - flipped label
            if args.LF:
                lf_args = argparse.Namespace(**vars(args))
                lf_args.LF = True
                lf_args.CF = True
                lf_args.TS = False
                lf_args.TA = False
                
                lf_signal_data, lf_afib_label = preprocess_signal(signal, lf_args, afib_label, tokenizer, freq_tokenizer)
                lf_time_attr, lf_freq_attr = get_integrated_gradients(model, device, lf_signal_data, lf_afib_label)
                
                visualize_dual_domain(
                    signal, lf_time_attr, lf_freq_attr, 
                    lf_signal_data['freqs'], lf_signal_data['times'], 
                    lf_signal_data['quantized_spec'],
                    lf_signal_data, key, lf_args, model, "LF"
                )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main() 