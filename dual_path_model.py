import torch
import torch.nn as nn
import math
from transformers import BigBirdModel, BigBirdConfig, LongformerModel, LongformerConfig


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function (same as in models.py)
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class DualPathECGModel(nn.Module):
    """Model that processes both time-domain and frequency-domain representations of ECG signals"""
    def __init__(self, time_model_config, freq_model_config, model_type='big', fusion_dim=768):
        """
        Initialize the dual-path model
        
        Args:
            time_model_config: Configuration for the time domain model
            freq_model_config: Configuration for the frequency domain model
            model_type: 'big' for BigBird, 'long' for Longformer
            fusion_dim: Dimension of the fusion layer
        """
        super().__init__()
        
        # Initialize time domain and frequency domain models
        if model_type == 'big':
            self.time_model = BigBirdModel(time_model_config)
            self.freq_model = BigBirdModel(freq_model_config)
        elif model_type == 'long':
            self.time_model = LongformerModel(time_model_config)
            self.freq_model = LongformerModel(freq_model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type
        self.time_hidden_size = time_model_config.hidden_size
        self.freq_hidden_size = freq_model_config.hidden_size
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.time_hidden_size + self.freq_hidden_size, fusion_dim),
            nn.LayerNorm(fusion_dim),
            NewGELUActivation()
        )
        
        # Output layers
        self.time_output = nn.Linear(fusion_dim, time_model_config.vocab_size)
        self.classifier = nn.Linear(fusion_dim, 2)  # For AFib classification (0 or 1)
        
        # Loss functions
        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.classifier_loss_fct = nn.CrossEntropyLoss()
    
    def forward(self, 
                time_input_ids, 
                time_attention_mask, 
                freq_input_ids, 
                freq_attention_mask, 
                time_labels=None,
                freq_global_attention_mask=None,
                time_global_attention_mask=None,
                class_labels=None,
                output_attentions=False,
                output_hidden_states=False):
        """
        Forward pass through the dual-path model
        
        Args:
            time_input_ids: Input token IDs for time domain
            time_attention_mask: Attention mask for time domain
            freq_input_ids: Input token IDs for frequency domain
            freq_attention_mask: Attention mask for frequency domain
            time_labels: Labels for masked tokens in time domain
            freq_global_attention_mask: Global attention mask for frequency domain (Longformer only)
            time_global_attention_mask: Global attention mask for time domain (Longformer only)
            class_labels: Classification labels (0 for non-AFib, 1 for AFib)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - mlm_loss: Masked language modeling loss
                - classification_loss: Classification loss
                - logits: Token prediction logits
                - classification_logits: Classification logits
                - attentions: Attention weights if output_attentions is True
                - hidden_states: Hidden states if output_hidden_states is True
        """
        # Process time domain
        if self.model_type == 'long' and time_global_attention_mask is not None:
            time_outputs = self.time_model(
                input_ids=time_input_ids,
                attention_mask=time_attention_mask,
                global_attention_mask=time_global_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        else:
            time_outputs = self.time_model(
                input_ids=time_input_ids,
                attention_mask=time_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        
        # Process frequency domain
        if self.model_type == 'long' and freq_global_attention_mask is not None:
            freq_outputs = self.freq_model(
                input_ids=freq_input_ids,
                attention_mask=freq_attention_mask,
                global_attention_mask=freq_global_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        else:
            freq_outputs = self.freq_model(
                input_ids=freq_input_ids,
                attention_mask=freq_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
        
        # Get sequence representations
        time_sequence = time_outputs.last_hidden_state
        freq_sequence = freq_outputs.last_hidden_state
        
        # Get CLS token for classification
        time_cls = time_sequence[:, 0, :]
        freq_cls = freq_sequence[:, 0, :]
        
        # Fuse CLS tokens for classification
        cls_fusion = torch.cat([time_cls, freq_cls], dim=-1)
        cls_fused_features = self.fusion(cls_fusion)
        classification_logits = self.classifier(cls_fused_features)
        
        # Align sequence lengths if needed for token prediction
        # We'll use the time domain sequence length as reference
        aligned_freq_sequence = freq_sequence
        if time_sequence.size(1) != freq_sequence.size(1):
            # If freq sequence is longer, truncate
            if freq_sequence.size(1) > time_sequence.size(1):
                aligned_freq_sequence = freq_sequence[:, :time_sequence.size(1), :]
            # If freq sequence is shorter, pad with zeros
            else:
                padding = torch.zeros(
                    freq_sequence.size(0), 
                    time_sequence.size(1) - freq_sequence.size(1),
                    freq_sequence.size(2),
                    device=freq_sequence.device
                )
                aligned_freq_sequence = torch.cat([freq_sequence, padding], dim=1)
        
        # Concatenate and fuse sequence features
        fused_sequence = torch.cat([time_sequence, aligned_freq_sequence], dim=-1)
        fused_features = self.fusion(fused_sequence)
        
        # Get token prediction logits (for MLM task)
        logits = self.time_output(fused_features)
        
        # Calculate losses
        loss = None
        mlm_loss = None
        classification_loss = None
        
        if time_labels is not None:
            # Calculate masked language modeling loss
            # Only consider positions where time_labels != -100
            active_loss = time_labels.view(-1) != -100
            active_logits = logits.view(-1, logits.size(-1))[active_loss]
            active_labels = time_labels.view(-1)[active_loss]
            mlm_loss = self.mlm_loss_fct(active_logits, active_labels)
        
        if class_labels is not None:
            # Calculate classification loss
            classification_loss = self.classifier_loss_fct(classification_logits, class_labels)
        
        # Combine losses
        if mlm_loss is not None and classification_loss is not None:
            loss = mlm_loss + classification_loss
        elif mlm_loss is not None:
            loss = mlm_loss
        elif classification_loss is not None:
            loss = classification_loss
        
        # Prepare output dictionary
        output_dict = {
            'loss': loss,
            'mlm_loss': mlm_loss,
            'classification_loss': classification_loss,
            'logits': logits,
            'classification_logits': classification_logits
        }
        
        # Add attention outputs if requested
        if output_attentions:
            output_dict['attentions'] = time_outputs.attentions
            if self.model_type == 'long':
                output_dict['global_attentions'] = time_outputs.global_attentions
        
        # Add hidden states if requested
        if output_hidden_states:
            output_dict['hidden_states'] = time_outputs.hidden_states
        
        return output_dict 