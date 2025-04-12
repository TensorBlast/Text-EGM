#!/bin/bash

# Script to automate the Text-EGM pipeline on a RunPod instance.
# It handles setup, preprocessing, training for 'big' and 'long' models,
# inference, integrated gradients, and generating zip archives for transfer
# using runpodctl.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
REPO_URL="https://github.com/TensorBlast/Text-EGM.git"
REPO_DIR="Text-EGM" # The directory name created by git clone
SCRIPT_DIR="$(pwd)" # Save the directory where the script is executed
LOG_DIR="${SCRIPT_DIR}/run_logs" # Make LOG_DIR an absolute path
PRETRAINED_EMB_DIR="pretrained_embeddings"
CHECKPOINT_BASE_DIR="runs/checkpoint"
ARCHIVE_DIR="${SCRIPT_DIR}/archives" # Directory for storing zip archives

# Detect if running in Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER=true
    echo "Docker environment detected."
fi

# --- Helper Functions ---

# Executes a command, logging stdout and stderr to a file and also displaying output
log_exec() {
    local log_file="$1"
    shift
    echo "--------------------------------------------------" | tee -a "$log_file"
    echo "Running command: $*" | tee -a "$log_file"
    echo "Timestamp: $(date)" | tee -a "$log_file"
    echo "--------------------------------------------------" | tee -a "$log_file"
    
    # Execute command, tee to log file and display to console
    # For Python commands, set PYTHONUNBUFFERED=1 to ensure output is not buffered
    local cmd="$*"
    if [[ "$cmd" == python* ]]; then
        # Set PYTHONUNBUFFERED for Python commands to prevent output buffering
        cmd="PYTHONUNBUFFERED=1 $cmd"
    fi
    
    if ! eval "$cmd" 2>&1 | tee -a "$log_file"; then
        local exit_code=$?
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
        echo "ERROR: Command failed with exit code $exit_code. See details above." | tee -a "$log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
        return $exit_code
    fi
    
    echo "--------------------------------------------------" | tee -a "$log_file"
    echo "Command finished: $*" | tee -a "$log_file"
    echo "Timestamp: $(date)" | tee -a "$log_file"
    echo "--------------------------------------------------" | tee -a "$log_file"
    return 0
}

# Creates a zip archive of specified files or directories
create_archive() {
    local archive_name="$1"
    local source_path="$2"
    local description="$3"
    local log_file="$4"
    local archive_path="${ARCHIVE_DIR}/${archive_name}.zip"
    
    echo "Creating archive: ${archive_name}.zip from ${source_path}" | tee -a "$log_file"
    
    # Create the archive directory if it doesn't exist
    mkdir -p "${ARCHIVE_DIR}"
    
    # If source is a directory with specific file patterns to include
    if [[ "$source_path" == *"PATTERN:"* ]]; then
        local dir_path=$(echo "$source_path" | cut -d':' -f2)
        local patterns=$(echo "$source_path" | cut -d':' -f3-)
        
        # Create a temporary directory for the files to zip
        local temp_dir="${ARCHIVE_DIR}/temp_${archive_name}"
        mkdir -p "$temp_dir"
        
        # Copy only the specified file patterns to the temp directory
        for pattern in $patterns; do
            find "$dir_path" -maxdepth 1 -name "$pattern" -exec cp {} "$temp_dir/" \;
        done
        
        # Zip the temp directory contents
        (cd "$temp_dir" && zip -r "$archive_path" .) >> "$log_file" 2>&1
        
        # Clean up the temp directory
        rm -rf "$temp_dir"
    else
        # Standard zip operation
        zip -r "$archive_path" "$source_path" >> "$log_file" 2>&1
    fi
    
    echo "Archive created: ${archive_path}" | tee -a "$log_file"
    return 0
}

# Sends a file using runpodctl and logs the receive command
send_with_runpodctl() {
    local archive_path="$1"
    local description="$2"
    local log_file="$3"
    
    echo "Sending archive: ${archive_path} (${description})" | tee -a "$log_file"
    
    # Send the file and capture the output
    local runpodctl_output=$(runpodctl send "$archive_path" 2>&1)
    echo "$runpodctl_output" | tee -a "$log_file"
    
    # Extract the receive code and log the command to use
    local receive_code=$(echo "$runpodctl_output" | grep -o "runpodctl receive [^ ]*" | tail -1)
    
    if [ -n "$receive_code" ]; then
        echo -e "\n\n============ IMPORTANT: RUNPODCTL RECEIVE COMMAND ============" | tee -a "$log_file"
        echo "To download ${description}, run this command on your local machine:" | tee -a "$log_file"
        echo "$receive_code" | tee -a "$log_file"
        echo "============================================================\n" | tee -a "$log_file"
    else
        echo "WARNING: Could not extract receive code from runpodctl output" | tee -a "$log_file"
    fi
    
    return 0
}

# --- Main Script ---

echo "=================================================="
echo "Starting Full Text-EGM Pipeline on RunPod Instance"
TOTAL_STEPS=16 # Update total number of steps
echo "=================================================="

# Create log and archive directories with absolute paths
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" # Ensure directories exist early
date > "$LOG_DIR/00_start_time.log"

# 1. System Prep
echo "[Step 1/$TOTAL_STEPS] System Preparation..." | tee "$LOG_DIR/01_prep.log"
{
    echo "Ensuring required tools are installed..."
    if ! command -v wget &> /dev/null || ! command -v git &> /dev/null || ! command -v find &> /dev/null || ! command -v zip &> /dev/null; then
        echo "One or more required tools (wget, git, find, zip) not found, attempting installation..."
        # Assuming Debian/Ubuntu based system
        apt-get update -y
        apt-get install -y wget git findutils zip
    else
        echo "Required tools found."
    fi
    
    echo "Checking for runpodctl..."
    if ! command -v runpodctl &> /dev/null; then
        echo "WARNING: runpodctl not found. This should be pre-installed on RunPod instances."
        echo "Script may fail when trying to send files."
    else
        echo "runpodctl found, version: $(runpodctl version 2>&1 || echo 'unknown')"
    fi
    
    echo "System preparation checks complete."
} 2>&1 | tee -a "$LOG_DIR/01_prep.log"
echo "System preparation finished."

# 2. Clone Repo and Setup
echo "[Step 2/$TOTAL_STEPS] Cloning Repository and Installing Requirements..." | tee "$LOG_DIR/02_setup.log"
if [ -d "$REPO_DIR" ]; then
    echo "Repository directory '$REPO_DIR' already exists. Skipping clone." | tee -a "$LOG_DIR/02_setup.log"
else
    log_exec "$LOG_DIR/02_setup.log" git clone -b dev "$REPO_URL"
fi
cd "$REPO_DIR"
# Ensure we are in the correct directory
echo "Current directory: $(pwd)" | tee -a "$LOG_DIR/02_setup.log"
if [[ "$(basename "$(pwd)")" != "$REPO_DIR" ]]; then
    echo "ERROR: Failed to change directory to $REPO_DIR" | tee -a "$LOG_DIR/02_setup.log"
    exit 1
fi
log_exec "$LOG_DIR/02_setup.log" pip install -r requirements_colab.txt
echo "Setup complete."

# 3. Preprocessing
echo "[Step 3/$TOTAL_STEPS] Running Preprocessing..." | tee "$LOG_DIR/03_preprocess.log"
log_exec "$LOG_DIR/03_preprocess.log" python preprocess/preprocess_intra.py
echo "Preprocessing finished."

# --- BigBird Model Pipeline ---
echo "--- Starting BigBird Pipeline ---"

# 4. Pretrain Embeddings (BigBird)
echo "[Step 4/$TOTAL_STEPS] Pretraining BigBird Embeddings..." | tee "$LOG_DIR/04_pretrain_big.log"
log_exec "$LOG_DIR/04_pretrain_big.log" python pretrain_embeddings.py --model big --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "BigBird Embedding Pretraining finished."

# 5. Archive and Send BigBird Embeddings
echo "[Step 5/$TOTAL_STEPS] Archiving and Sending BigBird Embeddings..." | tee "$LOG_DIR/05_archive_big_emb.log"
create_archive "Big_Embeddings" "./$PRETRAINED_EMB_DIR/big_embedding_weights.pt" "BigBird Embeddings" "$LOG_DIR/05_archive_big_emb.log"
send_with_runpodctl "${ARCHIVE_DIR}/Big_Embeddings.zip" "BigBird Embeddings" "$LOG_DIR/05_archive_big_emb.log"

# 6. Train BigBird (Default)
echo "[Step 6/$TOTAL_STEPS] Training BigBird (Default)..." | tee "$LOG_DIR/06_train_big_default.log"
set +e # Temporarily disable exit on error for Python script capture
TRAIN_BIG_DEFAULT_OUTPUT=$(python train.py --model big --epochs 20 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/06_train_big_default.log")
PYTHON_EXIT_CODE=$?
set -e # Re-enable exit on error
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/06_train_big_default.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
BIG_DEFAULT_DIR_REL=$(echo "$TRAIN_BIG_DEFAULT_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird default checkpoint directory from training output." | tee -a "$LOG_DIR/06_train_big_default.log"
    # Fallback is still useful if script output format changes unexpectedly
    BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/06_train_big_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/06_train_big_default.log"
    fi
fi
BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
echo "BigBird Default Training finished. Checkpoint directory: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/06_train_big_default.log"

# 7. Run Inference (BigBird Default)
echo "[Step 7/$TOTAL_STEPS] Running Inference (BigBird Default)..." | tee "$LOG_DIR/07_infer_big_default.log"
log_exec "$LOG_DIR/07_infer_big_default.log" python inference.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --batch 8 --mask 0.75
echo "Inference (BigBird Default) finished."

# 8. Run Integrated Gradients (BigBird Default) + Archive and Send
echo "[Step 8/$TOTAL_STEPS] Running Integrated Gradients (BigBird Default)..." | tee "$LOG_DIR/08_intgrad_big_default.log"
log_exec "$LOG_DIR/08_intgrad_big_default.log" python visualize/int_grad.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --CF --pre --n_steps 20
echo "Integrated Gradients (BigBird Default) finished."

# Create archive of IntGrad results and checkpoint
echo "[Step 8/$TOTAL_STEPS] Archiving and Sending BigBird Default Results..." | tee -a "$LOG_DIR/08_intgrad_big_default.log"
create_archive "Big_Default_IntGrad" "PATTERN:$BIG_DEFAULT_DIR_REL:*.png *.npy" "BigBird Default IntGrad Results" "$LOG_DIR/08_intgrad_big_default.log"
send_with_runpodctl "${ARCHIVE_DIR}/Big_Default_IntGrad.zip" "BigBird Default IntGrad Results" "$LOG_DIR/08_intgrad_big_default.log"
create_archive "Big_Default_Checkpoint" "$BIG_DEFAULT_DIR_REL" "BigBird Default Checkpoint" "$LOG_DIR/08_intgrad_big_default.log"
send_with_runpodctl "${ARCHIVE_DIR}/Big_Default_Checkpoint.zip" "BigBird Default Checkpoint" "$LOG_DIR/08_intgrad_big_default.log"

# 9. Train BigBird (Pretrained Embeddings)
echo "[Step 9/$TOTAL_STEPS] Training BigBird (Pretrained Embeddings)..." | tee "$LOG_DIR/09_train_big_embedding.log"
set +e
TRAIN_BIG_EMB_OUTPUT=$(python train.py --model big --epochs 20 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/big_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/09_train_big_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/09_train_big_embedding.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
BIG_EMBEDDING_DIR_REL=$(echo "$TRAIN_BIG_EMB_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird embedding checkpoint directory from training output." | tee -a "$LOG_DIR/09_train_big_embedding.log"
    # Fallback is still useful
    BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/09_train_big_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
    fi
fi
BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
echo "BigBird Pretrained Embedding Training finished. Checkpoint directory: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/09_train_big_embedding.log"

# 10. Run Inference (BigBird Embedding)
echo "[Step 10/$TOTAL_STEPS] Running Inference (BigBird Embedding)..." | tee "$LOG_DIR/10_infer_big_embedding.log"
log_exec "$LOG_DIR/10_infer_big_embedding.log" python inference.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --batch 8 --mask 0.75
echo "Inference (BigBird Embedding) finished."

# 11. Run Integrated Gradients (BigBird Embedding) + Archive and Send
echo "[Step 11/$TOTAL_STEPS] Running Integrated Gradients (BigBird Embedding)..." | tee "$LOG_DIR/11_intgrad_big_embedding.log"
log_exec "$LOG_DIR/11_intgrad_big_embedding.log" python visualize/int_grad.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --CF --pre --n_steps 20
echo "Integrated Gradients (BigBird Embedding) finished."

# Create archive of IntGrad results and checkpoint
echo "[Step 11/$TOTAL_STEPS] Archiving and Sending BigBird Embedding Results..." | tee -a "$LOG_DIR/11_intgrad_big_embedding.log"
create_archive "Big_Emb_IntGrad" "PATTERN:$BIG_EMBEDDING_DIR_REL:*.png *.npy" "BigBird Embedding IntGrad Results" "$LOG_DIR/11_intgrad_big_embedding.log"
send_with_runpodctl "${ARCHIVE_DIR}/Big_Emb_IntGrad.zip" "BigBird Embedding IntGrad Results" "$LOG_DIR/11_intgrad_big_embedding.log"
create_archive "Big_Emb_Checkpoint" "$BIG_EMBEDDING_DIR_REL" "BigBird Embedding Checkpoint" "$LOG_DIR/11_intgrad_big_embedding.log"
send_with_runpodctl "${ARCHIVE_DIR}/Big_Emb_Checkpoint.zip" "BigBird Embedding Checkpoint" "$LOG_DIR/11_intgrad_big_embedding.log"

# --- Longformer Model Pipeline ---
echo "--- Starting Longformer Pipeline ---"

# 12. Pretrain Embeddings (Longformer)
echo "[Step 12/$TOTAL_STEPS] Pretraining Longformer Embeddings..." | tee "$LOG_DIR/12_pretrain_long.log"
log_exec "$LOG_DIR/12_pretrain_long.log" python pretrain_embeddings.py --model long --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "Longformer Embedding Pretraining finished."

# Archive and Send Longformer Embeddings
echo "[Step 12/$TOTAL_STEPS] Archiving and Sending Longformer Embeddings..." | tee -a "$LOG_DIR/12_pretrain_long.log"
create_archive "Long_Embeddings" "./$PRETRAINED_EMB_DIR/long_embedding_weights.pt" "Longformer Embeddings" "$LOG_DIR/12_pretrain_long.log"
send_with_runpodctl "${ARCHIVE_DIR}/Long_Embeddings.zip" "Longformer Embeddings" "$LOG_DIR/12_pretrain_long.log"

# 13. Train Longformer (Default)
echo "[Step 13/$TOTAL_STEPS] Training Longformer (Default)..." | tee "$LOG_DIR/13_train_long_default.log"
set +e
TRAIN_LONG_DEFAULT_OUTPUT=$(python train.py --model long --epochs 20 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/13_train_long_default.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/13_train_long_default.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
LONG_DEFAULT_DIR_REL=$(echo "$TRAIN_LONG_DEFAULT_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer default checkpoint directory from training output." | tee -a "$LOG_DIR/13_train_long_default.log"
    # Fallback is still useful
    LONG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/13_train_long_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
    fi
fi
LONG_DEFAULT_CKPT_NAME=$(basename "$LONG_DEFAULT_DIR_REL")
echo "Longformer Default Training finished. Checkpoint directory: $LONG_DEFAULT_DIR_REL (Name: $LONG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/13_train_long_default.log"

# 14. Run Inference (Longformer Default)
echo "[Step 14/$TOTAL_STEPS] Running Inference (Longformer Default)..." | tee "$LOG_DIR/14_infer_long_default.log"
log_exec "$LOG_DIR/14_infer_long_default.log" python inference.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --batch 8 --mask 0.75
echo "Inference (Longformer Default) finished."

# 15. Run Integrated Gradients (Longformer Default) + Archive and Send
echo "[Step 15/$TOTAL_STEPS] Running Integrated Gradients (Longformer Default)..." | tee "$LOG_DIR/15_intgrad_long_default.log"
log_exec "$LOG_DIR/15_intgrad_long_default.log" python visualize/int_grad.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --CF --pre --n_steps 20
echo "Integrated Gradients (Longformer Default) finished."

# Create archive of IntGrad results and checkpoint
echo "[Step 15/$TOTAL_STEPS] Archiving and Sending Longformer Default Results..." | tee -a "$LOG_DIR/15_intgrad_long_default.log"
create_archive "Long_Default_IntGrad" "PATTERN:$LONG_DEFAULT_DIR_REL:*.png *.npy" "Longformer Default IntGrad Results" "$LOG_DIR/15_intgrad_long_default.log"
send_with_runpodctl "${ARCHIVE_DIR}/Long_Default_IntGrad.zip" "Longformer Default IntGrad Results" "$LOG_DIR/15_intgrad_long_default.log"
create_archive "Long_Default_Checkpoint" "$LONG_DEFAULT_DIR_REL" "Longformer Default Checkpoint" "$LOG_DIR/15_intgrad_long_default.log"
send_with_runpodctl "${ARCHIVE_DIR}/Long_Default_Checkpoint.zip" "Longformer Default Checkpoint" "$LOG_DIR/15_intgrad_long_default.log"

# 16. Train Longformer (Pretrained Embeddings)
echo "[Step 16/$TOTAL_STEPS] Training Longformer (Pretrained Embeddings)..." | tee "$LOG_DIR/16_train_long_embedding.log"
set +e
TRAIN_LONG_EMB_OUTPUT=$(python train.py --model long --epochs 20 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/long_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/16_train_long_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/16_train_long_embedding.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
LONG_EMBEDDING_DIR_REL=$(echo "$TRAIN_LONG_EMB_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")
if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer embedding checkpoint directory from training output." | tee -a "$LOG_DIR/16_train_long_embedding.log"
    # Fallback is still useful
    LONG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/16_train_long_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/16_train_long_embedding.log"
    fi
fi
LONG_EMBEDDING_CKPT_NAME=$(basename "$LONG_EMBEDDING_DIR_REL")
echo "Longformer Pretrained Embedding Training finished. Checkpoint directory: $LONG_EMBEDDING_DIR_REL (Name: $LONG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/16_train_long_embedding.log"

# 17. Run Inference (Longformer Embedding)
echo "[Step 17/$TOTAL_STEPS] Running Inference (Longformer Embedding)..." | tee "$LOG_DIR/17_infer_long_embedding.log"
log_exec "$LOG_DIR/17_infer_long_embedding.log" python inference.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --batch 8 --mask 0.75
echo "Inference (Longformer Embedding) finished."

# 18. Run Integrated Gradients (Longformer Embedding) + Archive and Send
echo "[Step 18/$TOTAL_STEPS] Running Integrated Gradients (Longformer Embedding)..." | tee "$LOG_DIR/18_intgrad_long_embedding.log"
log_exec "$LOG_DIR/18_intgrad_long_embedding.log" python visualize/int_grad.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --CF --pre --n_steps 20
echo "Integrated Gradients (Longformer Embedding) finished."

# Create archive of IntGrad results and checkpoint
echo "[Step 18/$TOTAL_STEPS] Archiving and Sending Longformer Embedding Results..." | tee -a "$LOG_DIR/18_intgrad_long_embedding.log"
create_archive "Long_Emb_IntGrad" "PATTERN:$LONG_EMBEDDING_DIR_REL:*.png *.npy" "Longformer Embedding IntGrad Results" "$LOG_DIR/18_intgrad_long_embedding.log"
send_with_runpodctl "${ARCHIVE_DIR}/Long_Emb_IntGrad.zip" "Longformer Embedding IntGrad Results" "$LOG_DIR/18_intgrad_long_embedding.log"
create_archive "Long_Emb_Checkpoint" "$LONG_EMBEDDING_DIR_REL" "Longformer Embedding Checkpoint" "$LOG_DIR/18_intgrad_long_embedding.log"
send_with_runpodctl "${ARCHIVE_DIR}/Long_Emb_Checkpoint.zip" "Longformer Embedding Checkpoint" "$LOG_DIR/18_intgrad_long_embedding.log"

# Archive and send all logs
echo "Archiving and sending all logs..." | tee "$LOG_DIR/99_final_logs.log"
create_archive "All_Logs" "$LOG_DIR" "All Run Logs" "$LOG_DIR/99_final_logs.log"
send_with_runpodctl "${ARCHIVE_DIR}/All_Logs.zip" "All Run Logs" "$LOG_DIR/99_final_logs.log"

echo "=================================================="
echo "Script finished."
date | tee "$LOG_DIR/99_end_time.log"
echo "=================================================="

# Exit with 0 if no errors caused an early exit (due to set -e or explicit exit)
exit 0 