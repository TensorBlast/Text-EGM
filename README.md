# Interpretation of Intracardiac Electrograms Through Textual Representations - Reproducing Paper for UIUC DLH 

DLH Final Project code for "[Interpretation of Intracardiac Electrograms Through Textual Representations](https://arxiv.org/abs/2402.01115)"

Team - pasi2
       sp108



## Set Up Environment

 `pip install -r requirements.txt`

## Set Up Data

1. Although the data we curated is not publicly available, we do have experimental results on an external dataset (main results are in Table 2 in the paper), namely the "Intracardiac Atrial Fibrillation Database" available on PhysioNet.

2. To set up this data, `cd` into the `preprocess` folder.

3. Please execute the following command to download the data.

```
wget https://physionet.org/static/published-projects/iafdb/intracardiac-atrial-fibrillation-database-1.0.0.zip
```

4. Unzip the file by executing

```
unzip intracardiac-atrial-fibrillation-database-1.0.0
```

5. Now execute the folllowing command to preprocess the data.

```
sh preprocess.sh
```

6. This should create a data folder with several `.npy` for training, validation, and test.

## Steps to Run Code

The following steps outline how to run the pipeline for the **BigBird** model using the PhysioNet dataset. These steps can be adapted for the **Longformer** model by replacing `--model big` with `--model long` in the commands. Default batch sizes are used below, but you can adjust them using `--batch <size>` (e.g., `--batch 8` for training, `--batch 16` for inference/pretraining).

### 1. Preprocessing (Already covered in Set Up Data)

Ensure you have run the preprocessing steps described in the "Set Up Data" section. The main command is:

```bash
cd preprocess
python preprocess_intra.py
cd ..
```

### 2. Pretrain Embeddings (Optional)

This step pretrains token embeddings before the main training task.

```bash
python pretrain_embeddings.py --model big --epochs 30 --output_dir ./pretrained_embeddings
```
The resulting embeddings will be saved in `./pretrained_embeddings/big_embedding_weights.pt`.

### 3. Training

You can train the model either from scratch or using the pretrained embeddings.

**a) Train from Scratch:**

```bash
python train.py --model big --epochs 20 --no-use_ce
```

**b) Train with Pretrained Embeddings:**

Make sure the embedding file from step 2 exists.

```bash
python train.py --model big --epochs 20 --no-use_ce --pretrained_embeddings ./pretrained_embeddings/big_embedding_weights.pt
```

Training checkpoints will be saved in a timestamped directory under `runs/checkpoint/`, e.g., `runs/checkpoint/saved_best_date_time_big_...`. Note the name of your specific checkpoint directory (`<CHECKPOINT_NAME>`) for the next steps.

### 4. Inference (Evaluation)

Run inference using a trained checkpoint. Replace `<CHECKPOINT_NAME>` with the actual directory name obtained from the training step.

```bash
python inference.py --checkpoint <CHECKPOINT_NAME> --model big --mask 0.75
```

### 5. Visualization (Integrated Gradients)

Generate integrated gradient visualizations for a trained checkpoint. Replace `<CHECKPOINT_NAME>` with the actual directory name obtained from the training step.

```bash
python visualize/int_grad.py --checkpoint <CHECKPOINT_NAME> --model big --CF --pre --n_steps 20
```

The visualizations (PNG files) and raw attribution scores (NPY files) will be saved within the corresponding checkpoint directory (`runs/checkpoint/<CHECKPOINT_NAME>/`).

Note: The original `run_pipeline.sh` script includes additional steps for downloading artifacts and logs, especially when running on cloud platforms. The commands above represent the core execution steps for the BigBird model. Repeat steps 2-5 with `--model long` for the Longformer variant.