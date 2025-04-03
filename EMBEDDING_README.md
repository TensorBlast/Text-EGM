# Pretrained Embeddings for ECG Signal Interpolation

This extension allows for pretraining and using custom embeddings for ECG signal interpolation, potentially improving the model's ability to reconstruct and interpret ECG signals.

## Overview

By default, the transformer models (BigBird and Longformer) use generic embeddings that aren't specifically optimized for ECG signals. This extension adds:

1. A new script (`pretrain_embeddings.py`) to train embeddings specifically for ECG signals
2. The ability to load these pretrained embeddings in both training and inference
3. Options to freeze or finetune these pretrained embeddings

## Pretraining Embeddings

To pretrain embeddings for ECG signals:

```bash
python pretrain_embeddings.py --model big --epochs 30 --batch 16 --output_dir ./pretrained_embeddings
```

Key parameters:
- `--model`: Model architecture (big, long, clin_bird, clin_long)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--output_dir`: Directory to save pretrained embeddings
- `--temperature`: Temperature for contrastive loss (default: 0.1)
- `--toy`: Use toy dataset for testing
- `--dry-run`: Use only 10% of data for quick testing

The pretraining script uses a contrastive learning objective to train embeddings that preserve the similarity between ECG signals. It saves the best embeddings based on validation loss.

## Using Pretrained Embeddings

### Training with Pretrained Embeddings

```bash
python train.py --model big --pretrained_embeddings ./pretrained_embeddings/big_embedding_weights.pt
```

To freeze the pretrained embeddings (prevent them from being updated during training):

```bash
python train.py --model big --pretrained_embeddings ./pretrained_embeddings/big_embedding_weights.pt --freeze_embeddings
```

### Inference with Pretrained Embeddings

```bash
python inference.py --model big --checkpoint your_checkpoint_name --pretrained_embeddings ./pretrained_embeddings/big_embedding_weights.pt
```

## Implementation Details

The embedding pretraining uses a contrastive learning objective where:
1. Similar signals should have similar embeddings
2. The model learns to distinguish between different signal patterns
3. Embeddings capture the temporal dependencies in ECG signals

The advantage of pretraining embeddings separately is that they can learn a good representation of ECG signals before the masked modeling task, potentially improving interpolation performance, especially for complex patterns.

## Workflow

1. Pretrain embeddings using `pretrain_embeddings.py`
2. Train the main model with pretrained embeddings using `train.py` with `--pretrained_embeddings`
3. Run inference with `inference.py` using the same pretrained embeddings

## Tips

- For best results, train embeddings on a large dataset of ECG signals
- Try both frozen and unfrozen embeddings to see which gives better performance
- The pretraining script includes early stopping to prevent overfitting
- Visualization of embedding training is saved in the output directory 