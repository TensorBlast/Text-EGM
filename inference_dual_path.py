import torch
torch.set_num_threads(2)
import numpy as np
from transformers import BigBirdTokenizer, LongformerTokenizer, BigBirdConfig, LongformerConfig
import argparse
from dual_domain_dataset import DualDomainECGDataset
from dual_path_model import DualPathECGModel
from frequency_tokenizer import FrequencyTokenizer
from torch.utils.data import DataLoader
import gc
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--signal_size', type=int, default=250, help='Signal size for quantization')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint directory')
    parser.add_argument('--model_file', type=str, default='best_model.pt', help='Model file name')
    parser.add_argument('--model', type=str, default='big', choices=['big', 'long'], help='Model type')
    parser.add_argument('--mask', type=float, default=0.75, help='Percentage to mask for signal')
    parser.add_argument('--TS', action='store_true', help='Enable Token Substitution')
    parser.add_argument('--TA', action='store_true', help='Enable Token Addition')
    parser.add_argument('--LF', action='store_true', help='Enable Label Flipping')
    parser.add_argument('--toy', action='store_true', help='Use toy dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--save_visualizations', action='store_true', help='Save visualizations of predictions')
    return parser.parse_args()

def create_toy(dataset, spec_ind):
    toy_dataset = {}
    for i in dataset.keys():
        _, placement, _, _ = i
        if placement in spec_ind:
            toy_dataset[i] = dataset[i]    
    return toy_dataset

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def get_prediction_metrics(original_signals, predicted_signals, mask_indices, keys):
    """Calculate prediction metrics for masked token reconstruction"""
    metrics = []
    
    for i in range(len(original_signals)):
        orig = original_signals[i]
        pred = predicted_signals[i]
        mask_idx = mask_indices[i]
        key = keys[i]
        
        # Calculate metrics only on masked tokens
        masked_orig = orig[mask_idx]
        masked_pred = pred[mask_idx]
        
        # Calculate error metrics
        mse = np.mean((masked_orig - masked_pred) ** 2)
        mae = np.mean(np.abs(masked_orig - masked_pred))
        rmse = np.sqrt(mse)
        
        # Store metrics
        metrics.append({
            'key': key,
            'mse': mse,
            'mae': mae,
            'rmse': rmse
        })
    
    return pd.DataFrame(metrics)

def visualize_prediction(signal, predicted_signal, mask_indices, key, args, domain="time"):
    """Visualize original signal, masked tokens, and predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot original signal
    plt.plot(signal, 'b-', alpha=0.5, label='Original Signal')
    
    # Plot masked tokens in the original signal
    plt.scatter(mask_indices, signal[mask_indices], color='red', marker='o', label='Masked Tokens (Original)')
    
    # Plot predictions for masked tokens
    plt.scatter(mask_indices, predicted_signal[mask_indices], color='green', marker='x', label='Predictions')
    
    plt.title(f'{domain.capitalize()} Domain Signal Reconstruction - {key}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(args.output_dir, f'{domain}_{key}_prediction.png')
    plt.savefig(output_path)
    plt.close()

def inference(model, tokenizer, data_loader, device, args):
    """Run inference with the dual-path model"""
    model.eval()
    
    # Initialize lists for storing results
    stitched_sequences = []
    ground_truth_sequences = []
    masked_positions_list = []
    MSEs_signals = []
    MAEs_signals = []
    ground_truth_afib = []
    pred_afib = []
    mean_accuracies_afib = []
    all_attentions = []
    count_afib = 0
    count_norm = 0
    all_global_attentions = []
    all_tokens = []
    count_index = 0
    count_index_list = []
    
    # Set up progress bar
    progress_bar = tqdm(data_loader, desc="Inference")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            time_input_ids = batch['time_input_ids'].to(device)
            freq_input_ids = batch['freq_input_ids'].to(device)
            time_attention_mask = batch['time_attention_mask'].to(device)
            freq_attention_mask = batch['freq_attention_mask'].to(device)
            time_labels = batch['time_labels'].to(device)
            time_mask = batch['time_mask'].to(device)
            keys = batch['key']
            raw_signal = batch['raw_signal'].to(device)
            
            # Get min and max values for denormalization
            time_min_val = batch['time_min_val'].numpy()
            time_max_val = batch['time_max_val'].numpy()
            
            # Handle global attention for Longformer
            time_global_attention_mask = batch.get('time_global_attention_mask')
            freq_global_attention_mask = batch.get('freq_global_attention_mask')
            if time_global_attention_mask is not None:
                time_global_attention_mask = time_global_attention_mask.to(device)
                freq_global_attention_mask = freq_global_attention_mask.to(device)
            
            # Forward pass with attention outputs
            outputs = model(
                time_input_ids=time_input_ids, 
                time_attention_mask=time_attention_mask,
                freq_input_ids=freq_input_ids, 
                freq_attention_mask=freq_attention_mask,
                time_labels=time_labels,
                time_global_attention_mask=time_global_attention_mask,
                freq_global_attention_mask=freq_global_attention_mask,
                output_attentions=True,
                output_hidden_states=True
            )
            
            # Get model predictions and attentions
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            attentions = outputs.get('attentions', None)
            global_attentions = outputs.get('global_attentions', None)
            
            # Add these print statements in the inference loop
            print("Raw predictions (first few):", predictions[0, :10])
            print("Unique prediction values:", torch.unique(predictions))
            print("Token distribution:", torch.bincount(predictions.flatten()))
            
            # Process each sample in the batch
            for i in range(time_input_ids.size(0)):
                # Get original and predicted token IDs for the current sample (skip special tokens)
                orig_tokens = time_labels[i, 1:-2].cpu().numpy()  # Skip CLS, SEP tokens
                pred_tokens = predictions[i, 1:-2].cpu().numpy()
                
                # Get mask indices for the current sample (skip special tokens)
                sample_mask = time_mask[i, 1:-2].cpu().numpy()
                
                # Convert token IDs to signal values
                orig_signal = np.array([int(tokenizer.convert_ids_to_tokens(int(t)).split('_')[1]) 
                                       for t in orig_tokens if tokenizer.convert_ids_to_tokens(int(t)).startswith('signal_')])
                
                pred_tokens_list = [tokenizer.convert_ids_to_tokens(int(t)) for t in pred_tokens]
                pred_signal = np.array([int(token.split('_')[1]) 
                                      for token in pred_tokens_list if token.startswith('signal_')])
                
                # Adjust arrays to same length if needed
                min_len = min(len(orig_signal), len(pred_signal))
                orig_signal = orig_signal[:min_len]
                pred_signal = pred_signal[:min_len]
                sample_mask = sample_mask[:min_len]
                
                # Denormalize signals back to z-score scale
                # First convert from [0,1] to [-10,10] range
                pred_signal_denorm = (pred_signal / args.signal_size) * 20 - 10
                
                # Store attention information if available
                if attentions is not None and int(keys[i][-1]) in [0, 1]:
                    if int(keys[i][-1]) == 0 and not count_norm > 3:
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in time_labels[i]]
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_tokens.append(tokens_cpu)
                        all_attentions.append(attentions_cpu)
                        if global_attentions is not None:
                            global_attentions_cpu = [attn.detach().cpu().numpy() for attn in global_attentions]
                            all_global_attentions.append(global_attentions_cpu)
                        count_norm += 1
                        count_index_list.append(count_index)
                    elif int(keys[i][-1]) == 1 and not count_afib > 3:
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in time_labels[i]]
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_tokens.append(tokens_cpu)
                        all_attentions.append(attentions_cpu)
                        if global_attentions is not None:
                            global_attentions_cpu = [attn.detach().cpu().numpy() for attn in global_attentions]
                            all_global_attentions.append(global_attentions_cpu)
                        count_afib += 1
                        count_index_list.append(count_index)
                
                # Process predictions and calculate metrics
                masked_positions_i = (sample_mask == 0)
                preds_masked_i = pred_signal_denorm[masked_positions_i]
                
                # Stitch sequences
                stitched_seq = np.copy(orig_signal)
                stitched_seq[masked_positions_i] = preds_masked_i
                
                # Store results
                masked_positions_list.append(np.where(masked_positions_i)[0])
                ground_truth_seq = raw_signal[i].cpu().numpy()
                stitched_sequences.append(stitched_seq[:1000])
                ground_truth_sequences.append(ground_truth_seq[:1000])
                
                # Process AFib predictions
                afib_pred = torch.argmax(outputs['classification_logits'][i]).item()
                afib_gt = int(keys[i][-1])  # Get AFib label from key tuple
                ground_truth_afib.append(afib_gt)
                pred_afib.append(afib_pred)
                
                # Calculate metrics
                mse_signal = mean_squared_error(stitched_seq, ground_truth_seq)
                mae_signal = mean_absolute_error(stitched_seq, ground_truth_seq)
                MSEs_signals.append(mse_signal)
                MAEs_signals.append(mae_signal)
                
                # Calculate AFib accuracy
                mean_acc_afib = accuracy_score([afib_pred], [afib_gt])
                mean_accuracies_afib.append(mean_acc_afib)
                
                # Visualize if requested
                if args.save_visualizations:
                    visualize_prediction(
                        ground_truth_seq, 
                        stitched_seq, 
                        np.where(masked_positions_i)[0], 
                        keys[i], 
                        args, 
                        "time"
                    )
            
            count_index += 1
    
    # Print summary metrics
    print("\nDual-Path Model Inference Results:")
    print('MSE for Signal Interpolation:', np.mean(MSEs_signals))
    print('MAE for Signal Interpolation:', np.mean(MAEs_signals))
    print("Average Accuracy for AFib:", np.mean(mean_accuracies_afib))
    print(f'Ground Truth Afib: {ground_truth_afib}')
    print(f'Pred Afib: {pred_afib}')
    
    # Calculate and print confusion matrix metrics
    cm = confusion_matrix(ground_truth_afib, pred_afib)
    print(f'Confusion Matrix: {cm}')
    
    try:
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]
        sensitivity = TP / float(TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / float(TN + FP) if (TN + FP) != 0 else 0
        npv = TN / float(TN + FN) if (TN + FN) != 0 else 0
        ppv = TP / float(TP + FP) if (TP + FP) != 0 else 0
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("NPV:", npv)
        print("PPV:", ppv)
    except:
        print('ravel error')
    
    # Save results
    np_save = {
        'masked_signals': masked_positions_list,
        'gt_signals': ground_truth_sequences,
        'pred_signals': stitched_sequences,
        'gt_afib': ground_truth_afib,
        'pred_afib': pred_afib,
        'attentions': all_attentions,
        'global_attentions': all_global_attentions,
        'tokens': all_tokens,
        'index': count_index_list
    }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'mse': MSEs_signals,
        'mae': MAEs_signals,
        'afib_accuracy': mean_accuracies_afib
    })
    metrics_path = os.path.join(args.output_dir, 'dual_path_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Save numpy arrays
    np.save(os.path.join(args.output_dir, 'dual_path_results.npy'), np_save)
    
    return metrics_df

def main():
    args = get_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join('./runs/inference', f"dual_path_{args.model}")
    ensure_directory_exists(args.output_dir)
    
    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2)
    
    # Set device
    device = None
    try:
        if args.device == 'cuda' or args.device.startswith('cuda:'):
            if torch.cuda.is_available():
                device = torch.device(args.device)
            else:
                print(f"Warning: {args.device} requested but CUDA is not available")
                device = torch.device('cpu')
        else:
            device = torch.device(args.device)
    except:
        print(f"Warning: Invalid device '{args.device}'. Trying CUDA...")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA as fallback")
        else:
            device = torch.device('cpu')
            print("Using CPU as fallback")
    
    print(f"Using device: {device}")
    
    # Load test data
    print('Loading test data...')
    test = np.load('../data/test_intra.npy', allow_pickle=True).item()
    
    if args.toy:
        test = create_toy(test, [18])
    
    # Load model checkpoint
    checkpoint_path = os.path.join(f'./runs/checkpoint/dual_path/{args.checkpoint}', args.model_file)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get saved args if available
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        # Override some parameters with saved ones
        args.signal_size = saved_args.signal_size
        args.model = saved_args.model
        print(f"Using model type '{args.model}' and signal size {args.signal_size} from checkpoint")
    
    # Create custom tokens
    print('Creating custom tokens...')
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
    
    # Initialize tokenizers
    print('Initializing tokenizers...')
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
    
    # Add custom tokens to tokenizer
    tokenizer.add_tokens(custom_tokens)
    
    # Update vocab size in configs
    time_config.vocab_size = len(tokenizer)
    freq_config.vocab_size = len(tokenizer)
    
    # Create model
    print('Initializing model...')
    model = DualPathECGModel(
        time_model_config=time_config,
        freq_model_config=freq_config,
        model_type=args.model,
        fusion_dim=768
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'Loaded model from {checkpoint_path}')
    
    # Create test dataset and data loader
    print('Creating test dataset and data loader...')
    test_dataset = DualDomainECGDataset(test, tokenizer, args=args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    
    # Run inference
    print('Running inference...')
    inference(model, tokenizer, test_loader, device, args)
    
    print('Inference completed!')

if __name__ == '__main__':
    main() 