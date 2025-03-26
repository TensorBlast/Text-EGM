import torch
import torch.nn as nn
import math
from torch_ecg.models import ECG_CRNN
from copy import deepcopy

class VITModel(nn.Module):
    def __init__( self, vit_model, vit_model_hs, num_labels):
        super().__init__()

        self.vit_model = vit_model
        self.num_labels = num_labels
        self.classifier = nn.Linear(vit_model_hs, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, x, y, mask):
        outputs = self.vit_model(x, bool_masked_pos = mask)
        seq_output = outputs.sequence_output
        img_loss = outputs.loss
        logits = self.classifier(seq_output[:, 0, :])
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        loss = img_loss + ce_loss

        return logits, loss

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class TimeSeriesModel(nn.Module):
    def __init__( self, ts_model, ts_model_hs, num_labels):
        super().__init__()

        self.ts_model = ts_model
        self.num_labels = num_labels
        self.embedding_projection = nn.Linear(1, ts_model_hs)
        self.dense = nn.Linear(ts_model_hs, ts_model_hs)
        self.classifier = nn.Linear(ts_model_hs, self.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.act = NewGELUActivation()
        
        
    def forward(self, x, signal_y, attention, class_y):
        x_projected = self.embedding_projection(x.unsqueeze(-1))
        outputs = self.ts_model(inputs_embeds = x_projected, attention_mask = attention, labels = signal_y, output_hidden_states = True)
        hidden_states = outputs.hidden_states
        mlm_loss = outputs.loss
        hidden_states = hidden_states[:, 0, :]
        out = self.dropout(hidden_states)
        out = self.dense(out)
        out = self.act(out)
        out = self.dropout(out)
        logits = self.classifier(out)
        ce_loss = self.loss_fct(logits.view(-1, self.num_labels), class_y.view(-1))
        loss = mlm_loss + ce_loss

        return logits, loss

class TorchECGWrapper(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout=0.2):
        super().__init__()
        # Initialize ECG_CRNN for AFib classification
        classes = ["Non-AFib", "AFib"]
        n_leads = 1
        
        # Configure model for intracardiac EGM data
        from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
        from torch_ecg.model_configs import ECG_CRNN_CONFIG
        
        # Create a custom configuration with safety measures
        config = deepcopy(ECG_CRNN_CONFIG)
        config = adjust_cnn_filter_lengths(config, fs=1000)
        
        # Enhance stability
        config.cnn.batch_norm = True
        config.cnn.dropout = dropout
        
        # Use appropriate filter lengths for intracardiac data
        config.cnn.filter_lengths = [11, 7, 5, 5]
        
        # Initialize model with the configuration
        self.model = ECG_CRNN(
            classes=classes,
            n_leads=n_leads,
            config=config
        )
        
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self, x, labels=None):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Safety check for extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        logits = self.model(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            
        return logits, loss
    