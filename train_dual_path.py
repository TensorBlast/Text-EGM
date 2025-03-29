import torch
torch.set_num_threads(2)
import numpy as np
from transformers import BigBirdConfig, LongformerConfig, BigBirdTokenizer, LongformerTokenizer
import argparse
from dual_domain_dataset import DualDomainECGDataset
from torch.utils.data import DataLoader
import gc
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from optim import ScheduledOptim, early_stopping
from dual_path_model import DualPathECGModel


def get_args():
    parser = argparse.ArgumentParser(description="Training script for dual-path ECG model")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--signal_size', type=int, default=250, help='Signal size for quantization')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training')
    parser.add_argument('--warmup', type=int, default=2000, help='Warmup steps for optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--model', type=str, default='big', choices=['big', 'long'], help='Model type')
    parser.add_argument('--mask', type=float, default=0.75, help='Percentage to mask for signal')
    parser.add_argument('--mlm_weight', type=float, default=1.0, help='Weight for MLM loss')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Weight for CE loss')
    parser.add_argument('--TS', action='store_true', help='Enable Token Substitution')
    parser.add_argument('--TA', action='store_true', help='Enable Token Addition')
    parser.add_argument('--LF', action='store_true', help='Enable Label Flipping')
    parser.add_argument('--toy', action='store_true', help='Use toy dataset')
    parser.add_argument('--output_dir', type=str, default='./runs/checkpoint/dual_path', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Use only 10% of data for testing')
    return parser.parse_args()


def create_toy(dataset, spec_ind):
    """Create a toy dataset with specific indices"""
    toy_dataset = {}
    for i in dataset.keys():
        _, placement, _, _ = i
        if placement in spec_ind:
            toy_dataset[i] = dataset[i]    
    return toy_dataset


def create_subset(dataset, percentage=0.1):
    """Creates a subset of the dataset with a given percentage of samples"""
    subset = {}
    keys = list(dataset.keys())
    # Calculate how many samples to keep
    n_samples = max(1, int(len(keys) * percentage))
    # Randomly select keys
    selected_keys = np.random.choice(keys, size=n_samples, replace=False)
    # Create subset with selected keys
    for key in selected_keys:
        subset[key] = dataset[key]
    return subset


def ensure_directory_exists(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def train_epoch(model, train_loader, optimizer, device, args):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_classification_loss = 0
    batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        time_input_ids = batch['time_input_ids'].to(device)
        freq_input_ids = batch['freq_input_ids'].to(device)
        time_attention_mask = batch['time_attention_mask'].to(device)
        freq_attention_mask = batch['freq_attention_mask'].to(device)
        time_labels = batch['time_labels'].to(device)
        afib_label = batch['afib_label'].to(device)
        
        # Handle global attention for Longformer
        time_global_attention_mask = batch.get('time_global_attention_mask')
        freq_global_attention_mask = batch.get('freq_global_attention_mask')
        if time_global_attention_mask is not None:
            time_global_attention_mask = time_global_attention_mask.to(device)
            freq_global_attention_mask = freq_global_attention_mask.to(device)
        
        # Prepare MLM labels: -100 for non-masked tokens
        mlm_labels = torch.full_like(time_labels, -100)
        mask_indices = (time_input_ids == model.time_model.config.mask_token_id)
        mlm_labels[mask_indices] = time_labels[mask_indices]
        
        # Forward pass
        outputs = model(
            time_input_ids=time_input_ids, 
            time_attention_mask=time_attention_mask,
            freq_input_ids=freq_input_ids, 
            freq_attention_mask=freq_attention_mask,
            time_labels=mlm_labels,
            time_global_attention_mask=time_global_attention_mask,
            freq_global_attention_mask=freq_global_attention_mask,
            class_labels=afib_label
        )
        
        # Apply loss weights
        loss = args.mlm_weight * outputs['mlm_loss'] + args.ce_weight * outputs['classification_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_mlm_loss += outputs['mlm_loss'].item()
        total_classification_loss += outputs['classification_loss'].item()
        batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'mlm_loss': outputs['mlm_loss'].item(),
            'class_loss': outputs['classification_loss'].item()
        })
    
    # Calculate average losses
    avg_loss = total_loss / batches
    avg_mlm_loss = total_mlm_loss / batches
    avg_classification_loss = total_classification_loss / batches
    
    return avg_loss, avg_mlm_loss, avg_classification_loss


def validate(model, val_loader, device, args):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_mlm_loss = 0
    total_classification_loss = 0
    total_correct = 0
    total_samples = 0
    batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            # Move data to device
            time_input_ids = batch['time_input_ids'].to(device)
            freq_input_ids = batch['freq_input_ids'].to(device)
            time_attention_mask = batch['time_attention_mask'].to(device)
            freq_attention_mask = batch['freq_attention_mask'].to(device)
            time_labels = batch['time_labels'].to(device)
            afib_label = batch['afib_label'].to(device)
            
            # Handle global attention for Longformer
            time_global_attention_mask = batch.get('time_global_attention_mask')
            freq_global_attention_mask = batch.get('freq_global_attention_mask')
            if time_global_attention_mask is not None:
                time_global_attention_mask = time_global_attention_mask.to(device)
                freq_global_attention_mask = freq_global_attention_mask.to(device)
            
            # Prepare MLM labels: -100 for non-masked tokens
            mlm_labels = torch.full_like(time_labels, -100)
            mask_indices = (time_input_ids == model.time_model.config.mask_token_id)
            mlm_labels[mask_indices] = time_labels[mask_indices]
            
            # Forward pass
            outputs = model(
                time_input_ids=time_input_ids, 
                time_attention_mask=time_attention_mask,
                freq_input_ids=freq_input_ids, 
                freq_attention_mask=freq_attention_mask,
                time_labels=mlm_labels,
                time_global_attention_mask=time_global_attention_mask,
                freq_global_attention_mask=freq_global_attention_mask,
                class_labels=afib_label
            )
            
            # Apply loss weights
            loss = args.mlm_weight * outputs['mlm_loss'] + args.ce_weight * outputs['classification_loss']
            
            # Track losses
            total_loss += loss.item()
            total_mlm_loss += outputs['mlm_loss'].item()
            total_classification_loss += outputs['classification_loss'].item()
            
            # Calculate classification accuracy
            predicted = torch.argmax(outputs['classification_logits'], dim=1)
            total_correct += (predicted == afib_label).sum().item()
            total_samples += afib_label.size(0)
            
            batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mlm_loss': outputs['mlm_loss'].item(),
                'class_loss': outputs['classification_loss'].item()
            })
    
    # Calculate average losses and accuracy
    avg_loss = total_loss / batches
    avg_mlm_loss = total_mlm_loss / batches
    avg_classification_loss = total_classification_loss / batches
    accuracy = total_correct / total_samples
    
    return avg_loss, avg_mlm_loss, avg_classification_loss, accuracy


def main():
    """Main training function"""
    args = get_args()
    
    # Create output directory
    output_dir = args.output_dir
    checkpoint_dir = os.path.join(output_dir, f"{args.model}_{args.lr}_{args.batch}_{args.mask}_{args.TS}_{args.TA}_{args.LF}")
    ensure_directory_exists(output_dir)
    ensure_directory_exists(checkpoint_dir)
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    
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
    
    # Load data
    print('Loading data...')
    train_data = np.load('../data/train_intra.npy', allow_pickle=True).item()
    val_data = np.load('../data/val_intra.npy', allow_pickle=True).item()
    
    # Create toy dataset if needed
    if args.toy:
        train_data = create_toy(train_data, [0, 1])
        val_data = create_toy(val_data, [14])
    
    # Create subset for dry run
    if args.dry_run:
        print('Dry run mode: using only 10% of the data')
        train_data = create_subset(train_data, 0.1)
        val_data = create_subset(val_data, 0.1)
    
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
    
    # Initialize model and tokenizer
    print('Initializing model...')
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
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    
    # Add custom tokens to tokenizer
    tokenizer.add_tokens(custom_tokens)
    
    # Update vocab size in configs
    time_config.vocab_size = len(tokenizer)
    freq_config.vocab_size = len(tokenizer)
    
    # Create model
    model = DualPathECGModel(
        time_model_config=time_config,
        freq_model_config=freq_config,
        model_type=args.model,
        fusion_dim=768
    ).to(device)
    
    # Create datasets and data loaders
    print('Creating dataset and data loader...')
    train_dataset = DualDomainECGDataset(train_data, tokenizer, args=args)
    val_dataset = DualDomainECGDataset(val_data, tokenizer, args=args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Create optimizer
    model_hidden_size = model.time_hidden_size  # Use time model hidden size
    optimizer = ScheduledOptim(
        Adam(filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-4, lr=args.lr, weight_decay=args.weight_decay),
        model_hidden_size, args.warmup
    )
    
    # Initialize early stopping
    early_stopper = early_stopping(patience=args.patience, verbose=True)
    
    # Training loop
    print('Starting training...')
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_mlm_loss, train_cls_loss = train_epoch(model, train_loader, optimizer, device, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_mlm_loss, val_cls_loss, val_accuracy = validate(model, val_loader, device, args)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, CLS: {train_cls_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, CLS: {val_cls_loss:.4f})")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'args': args,
            }, best_model_path)
            print(f"  Saved best model to {best_model_path}")
        
        # Early stopping
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Plot training progress
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, 'training_progress.png'))
    plt.close()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 