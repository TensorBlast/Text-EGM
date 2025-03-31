# Dual-Path ECG Model: Time and Frequency Domain Processing

This extension adds dual-path processing capabilities to the ECG signal interpolation model, allowing it to analyze both time and frequency domain representations simultaneously for improved performance and interpretability.

## Overview

The Dual-Path ECG Model processes electrocardiogram (ECG) signals in two complementary domains:

1. **Time Domain**: Captures the amplitude variations of the signal over time (traditional approach)
2. **Frequency Domain**: Captures the spectral components of the signal, revealing rhythmic patterns

By fusing information from both domains, the model can leverage complementary features that might be more evident in one domain than the other, potentially improving interpolation accuracy and enhancing model interpretability.

## Components

This extension consists of several new components:

1. `frequency_tokenizer.py`: Transforms ECG signals into frequency domain tokens using Short-Time Fourier Transform (STFT)
2. `dual_path_model.py`: Implements a model that processes both time and frequency domain representations and fuses them
3. `dual_domain_dataset.py`: Extends the dataset to handle both domains simultaneously
4. `train_dual_path.py`: Training script for the dual-path model
5. `visualize/dual_domain_viz.py`: Visualization tools to compare time and frequency domain attributions

## Usage

### Training

To train the dual-path model, use the `train_dual_path.py` script:

```bash
python train_dual_path.py --model big --mask 0.75 --batch 8 --epochs 10 --device cuda:0 --lr 1e-4 --warmup 1000 --signal_size 250 --output_dir ./runs/checkpoint/dual_path_big
```

Key parameters:
- `--model`: Model type ('big' for BigBird, 'long' for Longformer)
- `--mask`: Masking rate for masked language modeling
- `--TS`: Enable Token Substitution augmentation
- `--TA`: Enable Token Addition augmentation
- `--LF`: Enable Label Flipping augmentation

### Visualization and Counterfactual Analysis

For visualization and counterfactual analysis, use the `dual_domain_viz.py` script:

```bash
python visualize/dual_domain_viz.py --checkpoint ./runs/checkpoint/dual_path_big/big_1e-4_8_0.75_False_False_False --device cuda:0 --CF --TS --TA --LF --num_samples 5
```

This will:
1. Load a trained dual-path model
2. Visualize both time and frequency domain attributions
3. If `--CF` is enabled, perform counterfactual analyses with token substitution (`--TS`), token addition (`--TA`), and label flipping (`--LF`)




## Frequency-Domain Representation with Dual-Path Processing
The dual-path extension that processes both time and frequency domain representations of the ECG signal. This ablation is inspired by the paper - ["AnyECG: Foundational Models for Multitask Cardiac Analysis in Real-World Settings"](https://arxiv.org/abs/2411.17711)

## Concept
The current approach processes ECG signals primarily in the time domain by tokenizing amplitude values. However, cardiac arrhythmias often manifest distinctive frequency patterns that might not be optimally captured in time-domain representations alone. A dual-path model can process both time and frequency domain information simultaneously, potentially improving interpolation and interpretability.

## Benefits

The dual-path approach offers several advantages:

1. **Complementary Information**: Frequency domain analysis can reveal patterns that are not obvious in the time domain, such as subtle rhythmic abnormalities
2. **Improved Interpolation**: By using both domains, the model can better reconstruct masked segments of the signal
3. **Enhanced Interpretability**: Provides multi-perspective explanations of the model's decisions
4. **Noise Robustness**: Frequency domain representations can help distinguish between signal and noise
5. **Domain-Specific Insights**: Allows analysis of which domain (time or frequency) contributes more to specific predictions

## Implementation Details

### Frequency Tokenization

The frequency tokenizer converts ECG signals to the frequency domain using Short-Time Fourier Transform (STFT), which analyzes how frequency content changes over time. The resulting spectrogram is quantized and converted to tokens, similar to the time domain tokenization process.

### Dual Path Architecture

The model processes time and frequency domain tokens through separate transformer encoders, then fuses the representations before making predictions. This allows each domain to be processed optimally before combining their features.

### Visualization

The visualization tools show:
1. The original signal with time domain attributions
2. The frequency spectrogram
3. Frequency domain attributions mapped to the time-frequency plane

This provides a comprehensive view of what signal components (in both domains) are most important for the model's predictions.

## Commands for Finetuning and Counterfactual Analysis

### Step 1: Finetuning on Masked Signal

As we are working only with Afib patients and this ablation is focused on interpolation, we finetune either BigBird or LongFormer versions of the DualECGModel without any augmentation:

`python train_dual_path.py --mask 0.75 --batch 4 --lr 0.0001 --epochs 2 --dry-run --model long`

### Step 2: Evaluation 

To evaluate and save visualizations of the test data:
`python inference_dual_path.py --batch 1 --checkpoint <checkpoint_name>  --save_visualizations`


### Step 3: Visualization of Integrated Gradients
This step uses integrated gradients for feature attribution (which provides a more detailed understanding of which parts of the signal contribute most to predictions)
After finetuning, you'll need to run the visualization scripts with the counterfactual flag (--CF) enabled and different augmentation strategies. Here's how to do it for each counterfactual mode:

TA (for TS and LF use --TS and --LF flags, Note: TA + TS can be combined)

`python visualize/dual_domain_viz.py --model long --checkpoint <checkpoint_name> --TA --CF --output_dir ./cf_analysis/ta_only`


## Further Work

Potential extensions to this approach include:
1. Using more advanced time-frequency representations like wavelets
2. Adding more domains (e.g., wavelet domain)
3. Exploring different fusion strategies
4. Applying domain-specific augmentations

## Requirements

This extension requires additional dependencies:
- `scipy` for STFT computation
- `matplotlib` for visualization
- `captum` for integrated gradients calculation

## References

This approach is inspired by recent work in ECG analysis using multi-domain processing, including the papers "Interpretation of Intracardiac Electrograms Through Textual Representations" and "AnyECG: Foundational Models for Multitask Cardiac Analysis in Real-World Settings".