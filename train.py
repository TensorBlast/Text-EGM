import torch
torch.set_num_threads(2)
import numpy as np
from transformers import LongformerForMaskedLM, BigBirdForMaskedLM, BigBirdTokenizer, \
                                AutoModelForMaskedLM ,BigBirdConfig, AutoTokenizer, LongformerTokenizer, \
                                AutoImageProcessor, ViTForMaskedImageModeling, LongformerConfig
import argparse
from data_loader import EGMDataset, EGMIMGDataset, EGMTSDataset
from torch.utils.data import DataLoader
import gc
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from optim import ScheduledOptim, early_stopping
from models import VITModel, TimeSeriesModel
from runners import trainer, validate

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--signal_size', type = int, default = 250, help = 'Please choose the signal size')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'Please choose the type of device' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 100, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 2, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--model', type = str, default = 'big', help = 'Please choose which model to use')
    parser.add_argument('--use_ce', default=True, action=argparse.BooleanOptionalAction, help = 'Please choose whether to use CE loss or not')  
    parser.add_argument('--mask', type=float, default=0.75, help = 'Pleasee choose percentage to mask for signal')
    parser.add_argument('--mlm_weight', type = float, default = 1.0, help = 'Please choose the weight for the mlm loss')
    parser.add_argument('--ce_weight', type = float, default = 1.0, help = 'Please choose the weight for the ce loss')
    parser.add_argument('--TS', default=False, action=argparse.BooleanOptionalAction, help = 'Please choose whether to do Token Substitution')
    parser.add_argument('--TA', default=False, action=argparse.BooleanOptionalAction, help = 'Please choose whether to do Token Addition')
    parser.add_argument('--LF', default=False, action=argparse.BooleanOptionalAction, help = 'Please choose whether to do label flipping')
    parser.add_argument('--toy', default=False, action=argparse.BooleanOptionalAction, help = 'Please choose whether to use a toy dataset or not')
    parser.add_argument('--inference', default=False, action=argparse.BooleanOptionalAction, help = 'Please choose whether it is inference or not')
    parser.add_argument('--dry-run', default=False, action=argparse.BooleanOptionalAction, help = 'Use only 10% of data for quick testing')
    parser.add_argument('--pretrained_embeddings', type=str, default=None, help='Path to pretrained embeddings file')
    parser.add_argument('--freeze_embeddings', default=False, action=argparse.BooleanOptionalAction, help='Freeze pretrained embeddings during training')
    return parser.parse_args()
    
    
def create_toy(dataset, spec_ind):
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
    subset_keys = np.random.choice(len(keys), size=n_samples, replace=False)
    # Randomly select keys
    selected_keys = [keys[i] for i in subset_keys]
    # Create subset with selected keys
    for key in selected_keys:
        subset[key] = dataset[key]
    return subset

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def main():
    args = get_args()
    directory_path = f'./runs/checkpoint/saved_best_{args.lr}_{args.batch}_{args.patience}_{args.weight_decay}_{args.model}_{args.use_ce}_{args.mask}_{args.mlm_weight}_{args.ce_weight}_{args.toy}_{args.TS}_{args.TA}_{args.LF}{"_pretrained_emb" if args.pretrained_embeddings else ""}{"_frozen_emb" if args.freeze_embeddings else ""}{"_dry-run" if args.dry_run else ""}'
    ensure_directory_exists(directory_path)

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(2)
    
    # Add device fallback logic
    device = None
    try:
        if args.device == 'cuda' or args.device.startswith('cuda:'):
            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device(args.device)
            else:
                print(f"Warning: {args.device} requested but CUDA is not available")
                device = torch.device('cpu')
        else:
            # For 'cpu' or other devices
            device = torch.device(args.device)
    except:
        # If any error occurs (like invalid device string), try CUDA then fall back to CPU
        print(f"Warning: Invalid device '{args.device}'. Trying CUDA...")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("Using CUDA as fallback")
        else:
            device = torch.device('cpu')
            print("Using CPU as fallback")
    
    print(f"Using device: {device}")
    print('Loading Data...')
    print(f'CE being used: {args.use_ce}')
    
    train = np.load('../data/train_intra.npy', allow_pickle = True).item()
    val = np.load('../data/val_intra.npy', allow_pickle = True).item()
    
    if args.toy:
        train = create_toy(train, [0, 1])
        val = create_toy(val, [27])
    
    if args.dry_run:
        print('Dry run mode: using only 10% of the data')
        train = create_subset(train, 0.1)
        val = create_subset(val, 0.1)
    
    print('Creating Custom Tokens...')
    
    custom_tokens = [
        f"signal_{i}" for i in range(args.signal_size+1)
    ] + [
        f"afib_{i}" for i in range(2)
        ]
    if args.TA:
        custom_tokens += [
        f"augsig_{i}" for i in range(args.signal_size+1)
    ]
    
    print('Initalizing Model...')
    if args.model == 'big':
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
            
    if args.model == 'raw_big':
        configuration = BigBirdConfig(attention_type = 'original_full')
        model = BigBirdForMaskedLM(config = configuration).to(device)
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_bird':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_long':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model == 'vit':
        tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        pt_model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        args.num_patches = (pt_model.config.image_size // pt_model.config.patch_size) ** 2
        model_hidden_size = pt_model.config.hidden_size
        model = VITModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'big_ts':
        pt_model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        pt_model.config.attention_type = 'original_full'
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'long_ts':
        pt_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'long':
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
            
    if args.model == 'raw_long':
        configuration = LongformerConfig()
        model = LongformerForMaskedLM(config = configuration).to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size

    # Load pretrained embeddings if provided
    if args.pretrained_embeddings is not None:
        try:
            print(f"Loading pretrained embeddings from {args.pretrained_embeddings}")
            # Load the pretrained embeddings
            pretrained_embeds = torch.load(args.pretrained_embeddings, map_location=device)
            
            # For transformer models with direct embedding layers
            if args.model in ['big', 'raw_big', 'clin_bird', 'long', 'raw_long', 'clin_long']:
                embedding_layer = model.get_input_embeddings()
                # Load the weights for the embedding layer
                embedding_layer.load_state_dict(pretrained_embeds)
                
                # Freeze embeddings if specified
                if args.freeze_embeddings:
                    print("Freezing embedding layer")
                    embedding_layer.weight.requires_grad = False
                else:
                    print("Embeddings will be fine-tuned")
            
            # For TS models
            elif args.model in ['big_ts', 'long_ts']:
                embedding_layer = model.model.get_input_embeddings()
                # Load the weights for the embedding layer
                embedding_layer.load_state_dict(pretrained_embeds)
                
                # Freeze embeddings if specified
                if args.freeze_embeddings:
                    print("Freezing embedding layer")
                    embedding_layer.weight.requires_grad = False
                else:
                    print("Embeddings will be fine-tuned")
            
            print("Pretrained embeddings loaded successfully")
        except Exception as e:
            print(f"Error loading pretrained embeddings: {e}")
        
    print('Creating Dataset and DataLoader...')
    if args.model == 'vit':
        train_dataset = EGMIMGDataset(train, tokenizer, args = args)        
        val_dataset = EGMIMGDataset(val, tokenizer, args = args)
    elif args.model == 'big_ts' or args.model == 'long_ts':
        train_dataset = EGMTSDataset(train, args = args)        
        val_dataset = EGMTSDataset(val, args = args)
    else:
        train_dataset = EGMDataset(train, tokenizer, args = args)        
        val_dataset = EGMDataset(val, tokenizer, args = args)
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle = True)   
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle = True)
    
    optimizer = ScheduledOptim(
    Adam(filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-4, lr = args.lr, weight_decay=args.weight_decay), model_hidden_size, args.warmup)

    if args.use_ce:
        ce_loss = nn.CrossEntropyLoss(reduction = 'none')
    else:
        ce_loss = None

    train_losses = []
    val_losses = []
    all_epochs = []
    for epoch in range(args.epochs):
        
        all_epochs.append(epoch)
        train_loss = trainer(model, train_loader, optimizer, device, args, ce_loss)
        print(f"Training - Epoch: {epoch+1},Train Loss: {train_loss}")
        train_losses.append(train_loss)
        
        val_loss = validate(model, val_loader, device, args, ce_loss)
        print(f"Evaluation - Epoch: {epoch+1}, Val Loss: {val_loss}")
        val_losses.append(val_loss)
        
        model_state_dict = model.state_dict()
            
        checkpoint = {
            'model' : model_state_dict,
            'config_file' : 'config',
            'epoch' : epoch
        }
        
        if val_loss <= min(val_losses):
            torch.save(checkpoint, f'./{directory_path}/best_checkpoint.chkpt')
            print('    - [Info] The checkpoint file has been updated.')
        
        early_stop = early_stopping(val_losses, patience = args.patience, delta = 0.01)
    
        if early_stop:
            print('Validation loss has stopped decreasing. Early stopping...')
            break   
    
    fig1 = plt.figure('Figure 1')
    plt.plot(train_losses, label = 'train')
    plt.plot(val_losses, label= 'valid')
    plt.xlabel('epoch')
    plt.ylim([0.0, max(train_losses)])
    plt.ylabel('loss')
    plt.legend(loc ="upper right")
    plt.title('loss change curve')
    plt.savefig(f'./{directory_path}/loss_plot.png')
    plt.close()

if __name__ == '__main__':
    
    main()
