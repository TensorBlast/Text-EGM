import torch
torch.set_num_threads(2)
import numpy as np
from transformers import BigBirdForMaskedLM, LongformerForMaskedLM, BigBirdTokenizer, \
                        AutoModelForMaskedLM, LongformerTokenizer, AutoTokenizer, \
                        BigBirdConfig, LongformerConfig
import argparse
from data_loader import EGMDataset, EGMTSDataset
from torch.utils.data import DataLoader
import gc
from torch.optim import Adam
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from optim import ScheduledOptim, early_stopping

class EmbeddingPretrainingModel(nn.Module):
    """
    A model that only trains the embedding layer with a contrastive learning objective
    """
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.embedding = base_model.get_input_embeddings()
        
        # Non-linear projection head for better representation learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.hidden_size = hidden_size
        
    def forward(self, input_ids, positive_ids=None):
        # Get embeddings for input
        embeddings = self.embedding(input_ids)
        projected = self.projection(embeddings)
        
        # If we have positive examples (similar signals), compute contrastive loss
        if positive_ids is not None:
            positive_embeddings = self.embedding(positive_ids)
            positive_projected = self.projection(positive_embeddings)
            
            # Normalize embeddings for cosine similarity
            projected_norm = projected / projected.norm(dim=2, keepdim=True)
            positive_projected_norm = positive_projected / positive_projected.norm(dim=2, keepdim=True)
            
            # Compute similarity matrix
            similarity = torch.bmm(projected_norm, positive_projected_norm.transpose(1, 2))
            
            return embeddings, similarity, projected_norm, positive_projected_norm
        
        return embeddings

def get_args():
    parser = argparse.ArgumentParser(description="Pretrain embeddings for ECG signals")
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--signal_size', type=int, default=250, help='Signal size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--warmup', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--mask', type=float, default=0.4, help='Masking ratio')
    parser.add_argument('--model', type=str, default='big', help='Model type (big, long, etc.)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--hard_negatives', default=True, action=argparse.BooleanOptionalAction, help='Use hard negative mining')
    parser.add_argument('--neg_threshold', type=float, default=0.5, help='Threshold for hard negative mining')
    parser.add_argument('--toy', default=False, action=argparse.BooleanOptionalAction, help='Use toy dataset')
    parser.add_argument('--dry-run', default=False, action=argparse.BooleanOptionalAction, help='Use small subset for testing')
    parser.add_argument('--output_dir', type=str, default='./pretrained_embeddings', help='Output directory')
    parser.add_argument('--TS', default=False, action=argparse.BooleanOptionalAction, help='Token Substitution')
    parser.add_argument('--TA', default=False, action=argparse.BooleanOptionalAction, help='Token Addition')
    parser.add_argument('--LF', default=False, action=argparse.BooleanOptionalAction, help='Label Flipping')
    return parser.parse_args()

def create_toy(dataset, spec_ind):
    toy_dataset = {}
    
    for i in dataset.keys():
        _, placement, _, _ = i
        if placement in spec_ind:
            toy_dataset[i] = dataset[i]    
    return toy_dataset

def create_subset(dataset, percentage=0.1):
    subset = {}
    keys = list(dataset.keys())
    n_samples = max(1, int(len(keys) * percentage))
    subset_keys = np.random.choice(len(keys), size=n_samples, replace=False)
    selected_keys = [keys[i] for i in subset_keys]
    for key in selected_keys:
        subset[key] = dataset[key]
    return subset

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def hard_negative_mining(similarity, neg_threshold=0.5):
    """
    Identifies hard negatives in the similarity matrix
    Hard negatives are negatives (off-diagonal) that have high similarity
    """
    batch_size, seq_len, _ = similarity.size()
    
    # Create a mask for positive pairs (diagonal)
    diagonal_mask = torch.eye(seq_len, device=similarity.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create a mask for hard negative pairs (high similarity but not positive)
    hard_negative_mask = (similarity > neg_threshold) & (~diagonal_mask.bool())
    
    return hard_negative_mask

def contrastive_loss(similarity, temperature=0.1, hard_negatives=False, neg_threshold=0.5):
    """
    Computes contrastive loss (InfoNCE) for the similarity matrix
    With optional hard negative mining
    """
    batch_size, seq_len, _ = similarity.size()
    
    # Create labels - diagonal elements are positive pairs
    labels = torch.arange(seq_len, device=similarity.device).unsqueeze(0).repeat(batch_size, 1)
    
    # Scale logits by temperature
    logits = similarity / temperature
    
    # Apply hard negative mining if enabled
    if hard_negatives:
        # Find hard negatives (high similarity but not positive pairs)
        hard_negative_mask = hard_negative_mining(similarity, neg_threshold)
        
        # Apply higher weights to hard negatives
        hard_negative_weight = 3.0  # Weight for hard negatives
        hard_negative_logits = logits.clone()
        hard_negative_logits[hard_negative_mask] *= hard_negative_weight
        
        # Use weighted logits
        weighted_logits = hard_negative_logits
    else:
        weighted_logits = logits
    
    # Compute cross entropy loss
    loss = nn.CrossEntropyLoss()(weighted_logits.view(-1, seq_len), labels.view(-1))
    
    return loss

def train_embeddings(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0
    
    for batch_idx, (masked_sample, all_tokens, *rest) in enumerate(train_loader):
        # Move data to device
        masked_sample = masked_sample.to(device)
        all_tokens = all_tokens.to(device)
        
        # Forward pass
        _, similarity, _, _ = model(masked_sample, all_tokens)
        
        # Compute loss - contrastive learning objective with hard negative mining
        loss = contrastive_loss(
            similarity, 
            args.temperature, 
            args.hard_negatives, 
            args.neg_threshold
        )
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step_and_update_lr()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def validate_embeddings(model, val_loader, device, args):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (masked_sample, all_tokens, *rest) in enumerate(val_loader):
            # Move data to device
            masked_sample = masked_sample.to(device)
            all_tokens = all_tokens.to(device)
            
            # Forward pass
            _, similarity, _, _ = model(masked_sample, all_tokens)
            
            # Compute loss
            loss = contrastive_loss(
                similarity, 
                args.temperature, 
                args.hard_negatives, 
                args.neg_threshold
            )
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def visualize_embeddings(model, val_loader, device, epoch, output_dir):
    """
    Visualize the learned embeddings using t-SNE
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    # Only process a subset for visualization
    max_samples = 100
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (masked_sample, all_tokens, concatenated_sample, *rest) in enumerate(val_loader):
            if sample_count >= max_samples:
                break
                
            # Move data to device
            masked_sample = masked_sample.to(device)
            
            # Get embeddings (not projections)
            embeddings = model.embedding(masked_sample)
            
            # Take mean embedding for each sequence
            mean_embeddings = embeddings.mean(dim=1)
            
            # Get labels (last element of concatenated_sample is afib label)
            labels = concatenated_sample[:, -1].numpy()
            
            all_embeddings.append(mean_embeddings.cpu().numpy())
            all_labels.extend(labels)
            
            sample_count += masked_sample.size(0)
    
    if sample_count > 0:
        # Convert to numpy arrays
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)
        
        try:
            # Only import t-SNE if we're using it
            from sklearn.manifold import TSNE
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(all_embeddings)
            
            # Plot
            plt.figure(figsize=(10, 8))
            for label in np.unique(all_labels):
                idx = all_labels == label
                plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Label {int(label)}')
            
            plt.title(f'Embedding Visualization (Epoch {epoch})')
            plt.legend()
            plt.savefig(f"{output_dir}/embedding_viz_epoch_{epoch}.png")
            plt.close()
            
            print(f"Embedding visualization saved for epoch {epoch}")
        except ImportError:
            print("scikit-learn not available, skipping embedding visualization")
        except Exception as e:
            print(f"Error in embedding visualization: {e}")

def main():
    args = get_args()
    
    # Create output directory
    output_dir = args.output_dir
    ensure_directory_exists(output_dir)
    
    # Setup device
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
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA as fallback")
        else:
            device = torch.device('cpu')
            print("Using CPU as fallback")
    
    print(f"Using device: {device}")
    
    # Log training configuration
    print("\nTraining configuration:")
    print(f"  Model: {args.model}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Hard negative mining: {args.hard_negatives}")
    if args.hard_negatives:
        print(f"  Negative threshold: {args.neg_threshold}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch}")
    print(f"  Masking ratio: {args.mask}")
    print("\n")
    
    # Load data
    print("Loading data...")
    train_data = np.load('../data/train_intra.npy', allow_pickle=True).item()
    val_data = np.load('../data/val_intra.npy', allow_pickle=True).item()
    
    if args.toy:
        train_data = create_toy(train_data, [0, 1])
        val_data = create_toy(val_data, [27])
    
    if args.dry_run:
        print('Dry run mode: using only 10% of the data')
        train_data = create_subset(train_data, 0.1)
        val_data = create_subset(val_data, 0.1)
    
    # Create custom tokens
    print("Creating custom tokens...")
    custom_tokens = [
        f"signal_{i}" for i in range(args.signal_size+1)
    ] + [
        f"afib_{i}" for i in range(2)
    ]
    
    if args.TA:
        custom_tokens += [
            f"augsig_{i}" for i in range(args.signal_size+1)
        ]
    
    # Initialize tokenizer and base model
    print("Initializing model...")
    if args.model == 'big':
        base_model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
        base_model.config.attention_type = 'original_full'
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokenizer.add_tokens(custom_tokens)
        base_model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = base_model.config.hidden_size
    
    elif args.model == 'long':
        base_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        base_model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = base_model.config.hidden_size
    
    elif args.model == 'clin_bird':
        base_model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird")
        base_model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        tokenizer.add_tokens(custom_tokens)
        base_model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = base_model.config.hidden_size
    
    elif args.model == 'clin_long':
        base_model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        tokenizer.add_tokens(custom_tokens)
        base_model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = base_model.config.hidden_size
    
    # Create the embedding model
    model = EmbeddingPretrainingModel(base_model, model_hidden_size).to(device)
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = EGMDataset(train_data, tokenizer, args=args)
    val_dataset = EGMDataset(val_data, tokenizer, args=args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Setup optimizer
    optimizer = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-4, lr=args.lr, weight_decay=args.weight_decay),
        model_hidden_size, args.warmup
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    all_epochs = []

    
    for epoch in range(args.epochs):
        all_epochs.append(epoch)
        
        # Train
        train_loss = train_embeddings(model, train_loader, optimizer, device, args)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_embeddings(model, val_loader, device, args)
        print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        
        # Visualize embeddings every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            try:
                visualize_embeddings(model, val_loader, device, epoch, output_dir)
            except Exception as e:
                print(f"Error in embedding visualization: {e}")
        
        # Save model if it's the best so far
        if val_loss <= min(val_losses):
            embedding_weights = model.embedding.state_dict()
            torch.save(embedding_weights, f"{output_dir}/{args.model}_embedding_weights.pt")
            print(f"Saved embedding weights to {output_dir}/{args.model}_embedding_weights.pt")
        
        # Early stopping
        early_stop = early_stopping(val_losses, patience=args.patience, delta=0.01)
        if early_stop:
            print("Validation loss has stopped decreasing. Early stopping...")
            break
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(all_epochs, train_losses, label='Train Loss')
    plt.plot(all_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.close()
    
    print(f"Embedding pretraining completed! Weights saved to {output_dir}/{args.model}_embedding_weights.pt")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2)
    main() 