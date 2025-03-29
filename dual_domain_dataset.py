import torch
import numpy as np
from torch.utils.data import Dataset
from frequency_tokenizer import FrequencyTokenizer


class DualDomainECGDataset(Dataset):
    """Dataset for processing ECG data in both time and frequency domains"""
    
    def __init__(self, data_dict, time_tokenizer, args=None):
        """
        Initialize the dual domain dataset
        
        Args:
            data_dict: Dictionary containing ECG data
            time_tokenizer: Tokenizer for the time domain
            args: Arguments
        """
        self.data = list(data_dict.values())
        self.keys = list(data_dict.keys())
        self.args = args
        self.time_tokenizer = time_tokenizer
        self.freq_tokenizer = FrequencyTokenizer(
            n_freq_bins=50,
            signal_size=args.signal_size,
            sampling_rate=1000  # Adjust based on your ECG sampling rate
        )
        
        self.signal_size = args.signal_size
        self.vocab_size = len(self.time_tokenizer.get_vocab())
        self.cls = self.time_tokenizer.cls_token
        self.mask = self.time_tokenizer.mask_token
        self.sep = self.time_tokenizer.sep_token
        
        if self.args.TA:
            self.pad = self.time_tokenizer.pad_token
        
        self.curr_signal_len = 1000
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Returns:
            Tuple containing:
                - time_masked_sample: Masked time domain tokens
                - freq_masked_sample: Masked frequency domain tokens
                - time_tokens: Original time domain tokens
                - freq_tokens: Original frequency domain tokens
                - time_attention_mask: Attention mask for time domain
                - freq_attention_mask: Attention mask for frequency domain
                - time_mask: Mask for time domain tokens
                - freq_mask: Mask for frequency domain tokens
                - afib_label: AFib label
                - key: Sample key
        """
        sample = self.data[index]
        signal = sample[:1000]
        key = self.keys[index]
        afib_label = key[3]
        
        # Apply augmentations if enabled
        augmentation_scheme = np.random.randint(1, 5)
        if self.args.TS and augmentation_scheme == 2:
            signal = self.moving_average(signal)
        if self.args.LF and augmentation_scheme == 2:
            afib_label = self.label_flip(afib_label)
        
        # Get AFib token
        afib_token = f"afib_{int(afib_label)}"
        
        # Process time domain
        min_val, max_val = np.min(signal), np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        quantized_signal = np.floor(normalized_signal * self.signal_size).astype(int)
        quantized_signal_tokens = [f"signal_{i}" for i in quantized_signal]
        quantized_signal_ids = self.time_tokenizer.convert_tokens_to_ids(quantized_signal_tokens)
        
        # Process frequency domain
        freq_tokens, spec_min, spec_max, _, _, _ = self.freq_tokenizer.tokenize(signal)
        freq_token_ids = self.time_tokenizer.convert_tokens_to_ids(freq_tokens)
        
        # Get token IDs
        cls_id = self.time_tokenizer.convert_tokens_to_ids(self.cls)
        sep_id = self.time_tokenizer.convert_tokens_to_ids(self.sep)
        mask_id = self.time_tokenizer.convert_tokens_to_ids(self.mask)
        afib_id = self.time_tokenizer.convert_tokens_to_ids([afib_token])
        
        # Handle token addition if enabled
        if self.args.TA:
            # Time domain TA
            quantized_augsignal_tokens = [f"augsig_{i}" for i in quantized_signal]
            sampled_quantized_augsignal_tokens = self.sample_consecutive(
                quantized_augsignal_tokens, int(0.25 * len(quantized_signal_ids))
            )
            sampled_quantized_augsignal_ids = self.time_tokenizer.convert_tokens_to_ids(
                sampled_quantized_augsignal_tokens
            )
            pad_id = self.time_tokenizer.convert_tokens_to_ids(self.pad)
            
            # Create time domain tokens
            time_tokens = [cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
            if augmentation_scheme == 2:
                time_tokens2 = [cls_id] + quantized_signal_ids + sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]
            else:
                time_tokens2 = [cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
            
            # Frequency domain TA - we'll also add tokens to ensure both domains have similar structure
            freq_tokens_list = [cls_id] + freq_token_ids + [pad_id] * int(0.25 * len(freq_token_ids)) + [sep_id] + afib_id + [sep_id]
            if augmentation_scheme == 2:
                # Use same proportion of samples but from frequency domain
                sampled_freq_tokens = self.sample_consecutive(
                    freq_tokens, int(0.25 * len(freq_token_ids))
                )
                sampled_freq_token_ids = self.time_tokenizer.convert_tokens_to_ids(sampled_freq_tokens)
                freq_tokens_list2 = [cls_id] + freq_token_ids + sampled_freq_token_ids + [sep_id] + afib_id + [sep_id]
            else:
                freq_tokens_list2 = [cls_id] + freq_token_ids + [pad_id] * int(0.25 * len(freq_token_ids)) + [sep_id] + afib_id + [sep_id]
        else:
            # No token addition
            time_tokens = [cls_id] + quantized_signal_ids + [sep_id] + afib_id + [sep_id]
            time_tokens2 = time_tokens
            freq_tokens_list = [cls_id] + freq_token_ids + [sep_id] + afib_id + [sep_id]
            freq_tokens_list2 = freq_tokens_list
        
        # Create masks for time domain
        time_mask = np.ones_like(time_tokens)
        mask_indices_time = np.random.choice(
            self.curr_signal_len, int(self.args.mask * self.curr_signal_len), replace=False
        )
        time_mask[1:self.curr_signal_len+1][mask_indices_time] = 0
        time_mask[-2] = 0  # Always mask the AFib label
        
        # Create masks for frequency domain
        freq_mask = np.ones_like(freq_tokens_list)
        mask_indices_freq = np.random.choice(
            len(freq_token_ids), int(self.args.mask * len(freq_token_ids)), replace=False
        )
        freq_mask[1:len(freq_token_ids)+1][mask_indices_freq] = 0
        freq_mask[-2] = 0  # Always mask the AFib label
        
        # Create attention masks
        time_attention_mask = np.ones_like(time_tokens)
        freq_attention_mask = np.ones_like(freq_tokens_list)
        
        # Apply masks
        if self.args.TA:
            time_masked_sample = np.copy(time_tokens2)
            freq_masked_sample = np.copy(freq_tokens_list2)
        else:
            time_masked_sample = np.copy(time_tokens)
            freq_masked_sample = np.copy(freq_tokens_list)
            
        time_masked_sample[time_mask == 0] = mask_id
        freq_masked_sample[freq_mask == 0] = mask_id
        
        # Create global attention masks for Longformer (if needed)
        if self.args.model == 'long':
            time_global_attention_mask = np.zeros_like(time_tokens)
            time_global_attention_mask[0] = 1  # Global attention on CLS token
            time_global_attention_mask[-2] = 1  # Global attention on AFib token
            
            freq_global_attention_mask = np.zeros_like(freq_tokens_list)
            freq_global_attention_mask[0] = 1  # Global attention on CLS token
            freq_global_attention_mask[-2] = 1  # Global attention on AFib token
        else:
            time_global_attention_mask = None
            freq_global_attention_mask = None
        
        # Convert everything to tensors
        result = {
            'time_input_ids': torch.LongTensor(time_masked_sample),
            'freq_input_ids': torch.LongTensor(freq_masked_sample),
            'time_labels': torch.LongTensor(time_tokens),
            'freq_labels': torch.LongTensor(freq_tokens_list),
            'time_attention_mask': torch.tensor(time_attention_mask, dtype=torch.int),
            'freq_attention_mask': torch.tensor(freq_attention_mask, dtype=torch.int),
            'time_mask': torch.tensor(time_mask, dtype=torch.int),
            'freq_mask': torch.tensor(freq_mask, dtype=torch.int),
            'afib_label': torch.tensor(afib_label, dtype=torch.long),
            'key': key,
            'time_min_val': torch.tensor(min_val, dtype=torch.float32),
            'time_max_val': torch.tensor(max_val, dtype=torch.float32),
            'freq_min_val': torch.tensor(spec_min, dtype=torch.float32),
            'freq_max_val': torch.tensor(spec_max, dtype=torch.float32),
            'raw_signal': torch.tensor(signal, dtype=torch.float32)
        }
        
        if time_global_attention_mask is not None:
            result['time_global_attention_mask'] = torch.tensor(time_global_attention_mask, dtype=torch.int)
            result['freq_global_attention_mask'] = torch.tensor(freq_global_attention_mask, dtype=torch.int)
        
        return result
    
    def label_flip(self, afib_label):
        """Flip the AFib label (0 -> 1, 1 -> 0)"""
        if afib_label == 0:
            afib_label = 1
        elif afib_label == 1:
            afib_label = 0
        return afib_label
    
    def moving_average(self, signal, window_size=50):
        """Apply moving average smoothing"""
        return np.convolve(signal, np.ones(window_size), 'same') / window_size
    
    def sample_consecutive(self, signal, sample_size):
        """Sample consecutive elements from the signal"""
        max_start_index = len(signal) - sample_size
        start_index = np.random.randint(0, max_start_index)
        return signal[start_index:start_index + sample_size] 