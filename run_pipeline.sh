#!/bin/bash

# Script to automate the Text-EGM pipeline on a GCP instance.
# It handles setup, preprocessing, training for 'big' and 'long' models,
# inference, integrated gradients, optional Tailscale setup for connectivity,
# and parallel downloads of artifacts and logs to a local Mac.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
REPO_URL="https://github.com/TensorBlast/Text-EGM.git"
REPO_DIR="Text-EGM" # The directory name created by git clone
LOG_DIR="run_logs"
PRETRAINED_EMB_DIR="pretrained_embeddings"
CHECKPOINT_BASE_DIR="runs/checkpoint"
GCP_HOSTNAME="gcp-egm-runner" # Hostname for Tailscale

# --- Default Argument Values ---
USE_TAILSCALE=false
TAILSCALE_AUTH_KEY=""
MAC_USER=""
MAC_HOST=""
LOCAL_DEST_BASE_DIR=""

# --- Argument Parsing ---
# Simple argument parsing. Consider using getopt for more complex scenarios.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --use-tailscale)
        USE_TAILSCALE=true
        shift # past argument
        ;;
        --tailscale-auth-key)
        TAILSCALE_AUTH_KEY="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # Assume positional arguments: mac_user mac_host local_dest
        if [ -z "$MAC_USER" ]; then
            MAC_USER="$1"
        elif [ -z "$MAC_HOST" ]; then
            MAC_HOST="$1"
        elif [ -z "$LOCAL_DEST_BASE_DIR" ]; then
            LOCAL_DEST_BASE_DIR="$1"
        else
            echo "Unknown positional argument: $1"
            # Usage function might be better here
        fi
        shift # past argument
        ;;
    esac
done


# --- Validate Arguments ---
if [ -z "$MAC_USER" ] || [ -z "$MAC_HOST" ] || [ -z "$LOCAL_DEST_BASE_DIR" ]; then
    echo "Usage: $0 [--use-tailscale --tailscale-auth-key YOUR_TS_AUTH_KEY] <mac_user> <mac_host_or_ip_or_ts_name> <local_base_dest_dir>"
    echo "Example (no Tailscale): $0 myuser 192.168.1.100 /Users/myuser/gcp_results"
    echo "Example (with Tailscale): $0 --use-tailscale --tailscale-auth-key tskey-... myuser my-mac-tailscale-name /Users/myuser/gcp_results"
    echo ""
    echo "ERROR: Missing required arguments (mac_user, mac_host_or_ip_or_ts_name, local_base_dest_dir)."
    echo "If using --use-tailscale, --tailscale-auth-key is required, and <mac_host_or_ip_or_ts_name> should be the Mac's Tailscale name or IP."
    echo "Please ensure the following directories exist and are writable on the Mac at <local_base_dest_dir>:"
    echo "  - <local_base_dest_dir>/pretrained_embeddings"
    echo "  - <local_base_dest_dir>/runs/checkpoint"
    echo "  - <local_base_dest_dir>/int_grad_outputs" # New required directory
    echo "  - <local_base_dest_dir>/logs"
    echo "Also ensure passwordless SSH access (or Tailscale SSH) from this GCP instance to the Mac is configured."
    exit 1
fi

if $USE_TAILSCALE && [ -z "$TAILSCALE_AUTH_KEY" ]; then
    echo "ERROR: --tailscale-auth-key is required when using --use-tailscale."
    exit 1
fi

# Construct destination paths for clarity
LOCAL_EMB_DEST="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/pretrained_embeddings/"
LOCAL_CKPT_DEST_BASE="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/runs/checkpoint/"
LOCAL_INT_GRAD_DEST_BASE="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/int_grad_outputs/"
LOCAL_LOG_DEST="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/logs/"

# --- Helper Functions ---

# Executes a command, logging stdout and stderr to a file
log_exec() {
    local log_file="$1"
    shift
    echo "--------------------------------------------------" >> "$log_file"
    echo "Running command: $*" >> "$log_file"
    echo "Timestamp: $(date)" >> "$log_file"
    echo "--------------------------------------------------" >> "$log_file"
    # Execute command, appending stdout and stderr to the log file
    # Use eval to handle potential quotes in commands correctly, though be cautious
    if ! eval "$@" >> "$log_file" 2>&1; then
        local exit_code=$?
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> "$log_file"
        echo "ERROR: Command failed with exit code $exit_code. See details above." >> "$log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> "$log_file"
        echo "ERROR: Command failed: $*" >&2 # Also print to stderr
        # Set -e handles exiting, but log the error explicitly
        return $exit_code
    fi
     echo "--------------------------------------------------" >> "$log_file"
     echo "Command finished: $*" >> "$log_file"
     echo "Timestamp: $(date)" >> "$log_file"
     echo "--------------------------------------------------" >> "$log_file"
     # Ensure the function returns success if the command succeeded
     return 0
}

# Starts an scp process in the background to copy files/dirs
start_scp() {
    local scp_log_file="$1" # Log file for this specific scp operation
    local source_path="$2"
    local dest_path="$3"
    local description="$4"

    echo "$(date): Starting background SCP: $description ($source_path -> $dest_path)" | tee -a "$scp_log_file"
    # Run scp in the background, redirecting its output to the specific log
    # Add SSH options for non-interactive mode, useful with Tailscale/SSH keys
    scp -o BatchMode=yes -o StrictHostKeyChecking=no -r "$source_path" "$dest_path" >> "$scp_log_file" 2>&1 &
    echo $! # Return the PID of the background scp process
}

# Starts an scp process in the background specifically for copying a log file
start_log_scp() {
    local main_log_file="$1" # Log file generated by the main step
    local scp_log_file="$2"  # Log file for the scp operation itself
    local dest_path="$3"     # Destination on the Mac (should be the logs dir)
    local description="$4"

    echo "$(date): Starting background SCP for log: $description ($main_log_file -> $dest_path)" | tee -a "$scp_log_file"
    # Run scp in the background
    scp -o BatchMode=yes -o StrictHostKeyChecking=no "$main_log_file" "$dest_path" >> "$scp_log_file" 2>&1 &
    echo $! # Return the PID
}

# Starts an scp process in the background to copy integrated gradient results
start_int_grad_scp() {
    local scp_log_file="$1"
    local source_checkpoint_dir="$2" # e.g., runs/checkpoint/saved_best_...
    local dest_base_path="$3" # e.g., user@host:/path/to/int_grad_outputs/
    local description="$4"
    local ckpt_basename=$(basename "$source_checkpoint_dir")
    local final_dest_path="${dest_base_path}${ckpt_basename}/" # Append the specific dir name

    echo "$(date): Starting background SCP for IntGrad results: $description ($source_checkpoint_dir/*.png, *.npy -> $final_dest_path)" | tee -a "$scp_log_file"

    # Create the destination directory on the Mac first
    # Use ssh to run mkdir -p remotely. Ignore errors if it already exists.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${MAC_USER}@${MAC_HOST}" "mkdir -p '${LOCAL_DEST_BASE_DIR}/int_grad_outputs/${ckpt_basename}'" >> "$scp_log_file" 2>&1 || echo "Warning: mkdir -p on remote might have failed (possibly dir exists)" >> "$scp_log_file"

    # Use find to copy only specific file types, avoids copying the large checkpoint file again
    find "$source_checkpoint_dir" -maxdepth 1 \\( -name '*.png' -o -name '*.npy' \\) -exec \
        scp -o BatchMode=yes -o StrictHostKeyChecking=no {} "$final_dest_path" >> "$scp_log_file" 2>&1 \\; &

    echo $! # Return the PID of the background find/scp process
}

# --- Main Script ---

echo "=================================================="
echo "Starting Full Text-EGM Pipeline on GCP Instance"
TOTAL_STEPS=24 # Update total number of steps
if $USE_TAILSCALE; then
    echo "Tailscale Mode: ENABLED"
fi
echo "Target Mac: ${MAC_USER}@${MAC_HOST}"
echo "Destination Base Directory: ${LOCAL_DEST_BASE_DIR}"
echo "=================================================="
mkdir -p "$LOG_DIR" # Ensure log directory exists early
date > "$LOG_DIR/00_start_time.log"

# 1. System Prep
echo "[Step 1/$TOTAL_STEPS] System Preparation..." | tee "$LOG_DIR/01_prep.log"
{
    echo "Ensuring wget, git, and findutils are installed..."
    if ! command -v wget &> /dev/null || ! command -v git &> /dev/null || ! command -v find &> /dev/null; then
        echo "One or more required tools (wget, git, find) not found, attempting installation..."
        # Assuming Debian/Ubuntu based system
        sudo apt-get update -y
        sudo apt-get install -y wget git findutils
    else
        echo "wget, git, find found."
    fi

    if $USE_TAILSCALE; then
        echo "Ensuring Tailscale is installed..."
        if ! command -v tailscale &> /dev/null; then
            echo "Tailscale not found, installing..."
            curl -fsSL https://tailscale.com/install.sh | sh
        else
            echo "Tailscale found."
        fi
    fi
    echo "System preparation checks complete."
} >> "$LOG_DIR/01_prep.log" 2>&1
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

# 3. Optional: Activate Tailscale
if $USE_TAILSCALE; then
    echo "[Step 3/$TOTAL_STEPS] Activating Tailscale..." | tee "$LOG_DIR/03_tailscale_up.log"
    echo "Attempting to bring Tailscale up..." >> "$LOG_DIR/03_tailscale_up.log"
    # Check current status first
    if sudo tailscale status > /dev/null 2>&1; then
        echo "Tailscale already seems to be running/configured. Status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
        sudo tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
        echo "Attempting 'tailscale up' anyway to ensure correct key/settings... Use 'sudo tailscale logout' to reset if needed." | tee -a "$LOG_DIR/03_tailscale_up.log"
    fi
    # Use --accept-routes to potentially access Mac services, adjust hostname
    # Using --ssh enables Tailscale SSH server on the GCP instance (optional but useful)
    # Use quotes around the auth key in case it contains special characters
    log_exec "$LOG_DIR/03_tailscale_up.log" sudo tailscale up --authkey=\"${TAILSCALE_AUTH_KEY}\" --hostname=\"${GCP_HOSTNAME}\" --accept-routes --ssh
    echo "Tailscale activation attempt finished. Final status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
    sudo tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
    echo "Sleeping for 5 seconds to allow Tailscale connection to establish..." | tee -a "$LOG_DIR/03_tailscale_up.log"
    sleep 5 # Give Tailscale a moment to connect/update routes
else
    echo "[Step 3/$TOTAL_STEPS] Skipping Tailscale Activation." | tee "$LOG_DIR/03_tailscale_up.log"
fi

# 4. Preprocessing
echo "[Step 4/$TOTAL_STEPS] Running Preprocessing..." | tee "$LOG_DIR/04_preprocess.log"
log_exec "$LOG_DIR/04_preprocess.log" python preprocess/preprocess_intra.py
echo "Preprocessing finished."

# --- BigBird Model Pipeline ---
echo "--- Starting BigBird Pipeline ---"

# 5. Pretrain Embeddings (BigBird)
echo "[Step 5/$TOTAL_STEPS] Pretraining BigBird Embeddings..." | tee "$LOG_DIR/05_pretrain_big.log"
log_exec "$LOG_DIR/05_pretrain_big.log" python pretrain_embeddings.py --model big --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "BigBird Embedding Pretraining finished."

# 6. Download BigBird Embeddings (Background) + Log
echo "[Step 6/$TOTAL_STEPS] Downloading BigBird Embeddings and Log (background)..." | tee "$LOG_DIR/06_scp_big_emb.log"
PID_SCP_BIG_EMB_FILE=$(start_scp "$LOG_DIR/06_scp_big_emb.log" "./$PRETRAINED_EMB_DIR/big_embedding_weights.pt" "$LOCAL_EMB_DEST" "BigBird Embedding File")
PID_SCP_BIG_EMB_LOG=$(start_log_scp "$LOG_DIR/05_pretrain_big.log" "$LOG_DIR/06_scp_big_emb.log" "$LOCAL_LOG_DEST" "BigBird Pretrain Log")

# 7. Train BigBird (Default)
echo "[Step 7/$TOTAL_STEPS] Training BigBird (Default)..." | tee "$LOG_DIR/07_train_big_default.log"
set +e # Temporarily disable exit on error for Python script capture
TRAIN_BIG_DEFAULT_OUTPUT=$(python train.py --model big --epochs 50 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/07_train_big_default.log")
PYTHON_EXIT_CODE=$?
set -e # Re-enable exit on error
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/07_train_big_default.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
BIG_DEFAULT_DIR_REL=$(echo "$TRAIN_BIG_DEFAULT_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird default checkpoint directory from training output." | tee -a "$LOG_DIR/07_train_big_default.log"
    # Fallback is still useful if script output format changes unexpectedly
    BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/07_train_big_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
    fi
fi
BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
echo "BigBird Default Training finished. Checkpoint directory: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/07_train_big_default.log"

# 8. Download BigBird Default Checkpoint (Background) + Log
echo "[Step 8/$TOTAL_STEPS] Downloading BigBird Default Checkpoint and Log (background)..." | tee "$LOG_DIR/08_scp_big_default_ckpt.log"
PID_SCP_BIG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/08_scp_big_default_ckpt.log" "$BIG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Default Checkpoint")
PID_SCP_BIG_DEFAULT_LOG=$(start_log_scp "$LOG_DIR/07_train_big_default.log" "$LOG_DIR/08_scp_big_default_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Default Train Log")

# 9. Run Inference (BigBird Default)
echo "[Step 9/$TOTAL_STEPS] Running Inference (BigBird Default)..." | tee "$LOG_DIR/09_infer_big_default.log"
log_exec "$LOG_DIR/09_infer_big_default.log" python inference.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --batch 8 --mask 0.75
echo "Inference (BigBird Default) finished."

# 10. Run Integrated Gradients (BigBird Default) + Start Downloads
echo "[Step 10/$TOTAL_STEPS] Running Integrated Gradients (BigBird Default)..." | tee "$LOG_DIR/10_intgrad_big_default.log"
log_exec "$LOG_DIR/10_intgrad_big_default.log" python visualize/int_grad.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --CF --pre --n_steps 20
echo "Integrated Gradients (BigBird Default) finished."
# Start Downloads for IntGrad and previous Inference log
echo "[Step 10/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (BigBird Default) (background)..." | tee -a "$LOG_DIR/10_intgrad_big_default.log"
PID_SCP_BIG_DEFAULT_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/10_intgrad_big_default.log" "$BIG_DEFAULT_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "BigBird Default IntGrad Results")
PID_SCP_BIG_DEFAULT_INFER_LOG=$(start_log_scp "$LOG_DIR/09_infer_big_default.log" "$LOG_DIR/10_intgrad_big_default.log" "$LOCAL_LOG_DEST" "BigBird Default Inference Log")

# 11. Train BigBird (Pretrained Embeddings)
echo "[Step 11/$TOTAL_STEPS] Training BigBird (Pretrained Embeddings)..." | tee "$LOG_DIR/11_train_big_embedding.log"
set +e
TRAIN_BIG_EMB_OUTPUT=$(python train.py --model big --epochs 50 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/big_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/11_train_big_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/11_train_big_embedding.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
BIG_EMBEDDING_DIR_REL=$(echo "$TRAIN_BIG_EMB_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird embedding checkpoint directory from training output." | tee -a "$LOG_DIR/11_train_big_embedding.log"
    # Fallback is still useful
    BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/11_train_big_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/11_train_big_embedding.log"
    fi
fi
BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
echo "BigBird Pretrained Embedding Training finished. Checkpoint directory: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/11_train_big_embedding.log"

# 12. Download BigBird Embedding Checkpoint (Background) + Logs
echo "[Step 12/$TOTAL_STEPS] Downloading BigBird Embedding Checkpoint and Logs (background)..." | tee "$LOG_DIR/12_scp_big_embedding_ckpt.log"
PID_SCP_BIG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$BIG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Embedding Checkpoint")
PID_SCP_BIG_EMBEDDING_TRAIN_LOG=$(start_log_scp "$LOG_DIR/11_train_big_embedding.log" "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Embedding Train Log")
PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/10_intgrad_big_default.log" "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Default IntGrad Log")

# 13. Run Inference (BigBird Embedding)
echo "[Step 13/$TOTAL_STEPS] Running Inference (BigBird Embedding)..." | tee "$LOG_DIR/13_infer_big_embedding.log"
log_exec "$LOG_DIR/13_infer_big_embedding.log" python inference.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --batch 8 --mask 0.75
echo "Inference (BigBird Embedding) finished."

# 14. Run Integrated Gradients (BigBird Embedding) + Start Downloads
echo "[Step 14/$TOTAL_STEPS] Running Integrated Gradients (BigBird Embedding)..." | tee "$LOG_DIR/14_intgrad_big_embedding.log"
log_exec "$LOG_DIR/14_intgrad_big_embedding.log" python visualize/int_grad.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --CF --pre --n_steps 20
echo "Integrated Gradients (BigBird Embedding) finished."
# Start Downloads for IntGrad and previous Inference log
echo "[Step 14/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (BigBird Embedding) (background)..." | tee -a "$LOG_DIR/14_intgrad_big_embedding.log"
PID_SCP_BIG_EMBEDDING_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/14_intgrad_big_embedding.log" "$BIG_EMBEDDING_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "BigBird Embedding IntGrad Results")
PID_SCP_BIG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/13_infer_big_embedding.log" "$LOG_DIR/14_intgrad_big_embedding.log" "$LOCAL_LOG_DEST" "BigBird Embedding Inference Log")

# --- Longformer Model Pipeline ---
echo "--- Starting Longformer Pipeline ---"

# 15. Pretrain Embeddings (Longformer)
echo "[Step 15/$TOTAL_STEPS] Pretraining Longformer Embeddings..." | tee "$LOG_DIR/15_pretrain_long.log"
log_exec "$LOG_DIR/15_pretrain_long.log" python pretrain_embeddings.py --model long --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "Longformer Embedding Pretraining finished."

# 16. Download Longformer Embeddings (Background) + Logs
echo "[Step 16/$TOTAL_STEPS] Downloading Longformer Embeddings and Logs (background)..." | tee "$LOG_DIR/16_scp_long_emb.log"
PID_SCP_LONG_EMB_FILE=$(start_scp "$LOG_DIR/16_scp_long_emb.log" "./$PRETRAINED_EMB_DIR/long_embedding_weights.pt" "$LOCAL_EMB_DEST" "Longformer Embedding File")
PID_SCP_LONG_PRETRAIN_LOG=$(start_log_scp "$LOG_DIR/15_pretrain_long.log" "$LOG_DIR/16_scp_long_emb.log" "$LOCAL_LOG_DEST" "Longformer Pretrain Log")
PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/14_intgrad_big_embedding.log" "$LOG_DIR/16_scp_long_emb.log" "$LOCAL_LOG_DEST" "BigBird Embedding IntGrad Log") # Previous step's log

# 17. Train Longformer (Default)
echo "[Step 17/$TOTAL_STEPS] Training Longformer (Default)..." | tee "$LOG_DIR/17_train_long_default.log"
set +e
TRAIN_LONG_DEFAULT_OUTPUT=$(python train.py --model long --epochs 50 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/17_train_long_default.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/17_train_long_default.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
LONG_DEFAULT_DIR_REL=$(echo "$TRAIN_LONG_DEFAULT_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")

if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer default checkpoint directory from training output." | tee -a "$LOG_DIR/17_train_long_default.log"
    # Fallback is still useful
    LONG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/17_train_long_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/17_train_long_default.log"
    fi
fi
LONG_DEFAULT_CKPT_NAME=$(basename "$LONG_DEFAULT_DIR_REL")
echo "Longformer Default Training finished. Checkpoint directory: $LONG_DEFAULT_DIR_REL (Name: $LONG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/17_train_long_default.log"

# 18. Download Longformer Default Checkpoint (Background) + Log
echo "[Step 18/$TOTAL_STEPS] Downloading Longformer Default Checkpoint and Log (background)..." | tee "$LOG_DIR/18_scp_long_default_ckpt.log"
PID_SCP_LONG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/18_scp_long_default_ckpt.log" "$LONG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Default Checkpoint")
PID_SCP_LONG_DEFAULT_TRAIN_LOG=$(start_log_scp "$LOG_DIR/17_train_long_default.log" "$LOG_DIR/18_scp_long_default_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Default Train Log")

# 19. Run Inference (Longformer Default)
echo "[Step 19/$TOTAL_STEPS] Running Inference (Longformer Default)..." | tee "$LOG_DIR/19_infer_long_default.log"
log_exec "$LOG_DIR/19_infer_long_default.log" python inference.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --batch 8 --mask 0.75
echo "Inference (Longformer Default) finished."

# 20. Run Integrated Gradients (Longformer Default) + Start Downloads
echo "[Step 20/$TOTAL_STEPS] Running Integrated Gradients (Longformer Default)..." | tee "$LOG_DIR/20_intgrad_long_default.log"
log_exec "$LOG_DIR/20_intgrad_long_default.log" python visualize/int_grad.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --CF --pre --n_steps 20
echo "Integrated Gradients (Longformer Default) finished."
# Start Downloads for IntGrad and previous Inference log
echo "[Step 20/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (Longformer Default) (background)..." | tee -a "$LOG_DIR/20_intgrad_long_default.log"
PID_SCP_LONG_DEFAULT_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/20_intgrad_long_default.log" "$LONG_DEFAULT_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "Longformer Default IntGrad Results")
PID_SCP_LONG_DEFAULT_INFER_LOG=$(start_log_scp "$LOG_DIR/19_infer_long_default.log" "$LOG_DIR/20_intgrad_long_default.log" "$LOCAL_LOG_DEST" "Longformer Default Inference Log")

# 21. Train Longformer (Pretrained Embeddings)
echo "[Step 21/$TOTAL_STEPS] Training Longformer (Pretrained Embeddings)..." | tee "$LOG_DIR/21_train_long_embedding.log"
set +e
TRAIN_LONG_EMB_OUTPUT=$(python train.py --model long --epochs 50 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/long_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/21_train_long_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/21_train_long_embedding.log"
    exit $PYTHON_EXIT_CODE
fi
# Extract directory name regardless of whether it was created or already existed
LONG_EMBEDDING_DIR_REL=$(echo "$TRAIN_LONG_EMB_OUTPUT" | grep -Eo 'Directory (created|already exists): \./[^ ]+' | sed -E 's/Directory (created|already exists): \.\///' || echo "NOT_FOUND")
if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer embedding checkpoint directory from training output." | tee -a "$LOG_DIR/21_train_long_embedding.log"
    # Fallback is still useful
    LONG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/21_train_long_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/21_train_long_embedding.log"
    fi
fi
LONG_EMBEDDING_CKPT_NAME=$(basename "$LONG_EMBEDDING_DIR_REL")
echo "Longformer Pretrained Embedding Training finished. Checkpoint directory: $LONG_EMBEDDING_DIR_REL (Name: $LONG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/21_train_long_embedding.log"

# 22. Download Longformer Embedding Checkpoint (Background) + Logs
echo "[Step 22/$TOTAL_STEPS] Downloading Longformer Embedding Checkpoint and Logs (background)..." | tee "$LOG_DIR/22_scp_long_embedding_ckpt.log"
PID_SCP_LONG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LONG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Embedding Checkpoint")
PID_SCP_LONG_EMBEDDING_TRAIN_LOG=$(start_log_scp "$LOG_DIR/21_train_long_embedding.log" "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Embedding Train Log")
PID_SCP_LONG_DEFAULT_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/20_intgrad_long_default.log" "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Default IntGrad Log")

# 23. Run Inference (Longformer Embedding)
echo "[Step 23/$TOTAL_STEPS] Running Inference (Longformer Embedding)..." | tee "$LOG_DIR/23_infer_long_embedding.log"
log_exec "$LOG_DIR/23_infer_long_embedding.log" python inference.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --batch 8 --mask 0.75
echo "Inference (Longformer Embedding) finished."

# 24. Run Integrated Gradients (Longformer Embedding) + Start Downloads
echo "[Step 24/$TOTAL_STEPS] Running Integrated Gradients (Longformer Embedding)..." | tee "$LOG_DIR/24_intgrad_long_embedding.log"
log_exec "$LOG_DIR/24_intgrad_long_embedding.log" python visualize/int_grad.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --CF --pre --n_steps 20
echo "Integrated Gradients (Longformer Embedding) finished."
# Start Downloads for IntGrad and previous Inference log
echo "[Step 24/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (Longformer Embedding) (background)..." | tee -a "$LOG_DIR/24_intgrad_long_embedding.log"
PID_SCP_LONG_EMBEDDING_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/24_intgrad_long_embedding.log" "$LONG_EMBEDDING_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "Longformer Embedding IntGrad Results")
PID_SCP_LONG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/23_infer_long_embedding.log" "$LOG_DIR/24_intgrad_long_embedding.log" "$LOCAL_LOG_DEST" "Longformer Embedding Inference Log")

# --- Wait for all background downloads ---
echo "--- Waiting for background downloads to complete ---" | tee "$LOG_DIR/99_wait_downloads.log"
FAIL_COUNT=0
PIDS_TO_WAIT=(
    $PID_SCP_BIG_EMB_FILE
    $PID_SCP_BIG_EMB_LOG
    $PID_SCP_BIG_DEFAULT_CKPT
    $PID_SCP_BIG_DEFAULT_LOG
    $PID_SCP_BIG_DEFAULT_INFER_LOG # New
    $PID_SCP_BIG_DEFAULT_INTGRAD_FILES # New
    $PID_SCP_BIG_DEFAULT_INTGRAD_LOG # New (log for the scp of the default train log) - Renamed/Reused log file
    $PID_SCP_BIG_EMBEDDING_CKPT
    $PID_SCP_BIG_EMBEDDING_TRAIN_LOG # Renamed
    $PID_SCP_BIG_EMBEDDING_INFER_LOG # New
    $PID_SCP_BIG_EMBEDDING_INTGRAD_FILES # New
    $PID_SCP_BIG_EMBEDDING_INTGRAD_LOG # New (log for the scp of the embedding train log) - Renamed/Reused log file

    $PID_SCP_LONG_EMB_FILE
    $PID_SCP_LONG_PRETRAIN_LOG # Renamed
    $PID_SCP_LONG_DEFAULT_CKPT
    $PID_SCP_LONG_DEFAULT_TRAIN_LOG # Renamed
    $PID_SCP_LONG_DEFAULT_INFER_LOG # New
    $PID_SCP_LONG_DEFAULT_INTGRAD_FILES # New
    $PID_SCP_LONG_DEFAULT_INTGRAD_LOG # New

    $PID_SCP_LONG_EMBEDDING_CKPT
    $PID_SCP_LONG_EMBEDDING_TRAIN_LOG # Renamed
    $PID_SCP_LONG_EMBEDDING_INFER_LOG # New
    $PID_SCP_LONG_EMBEDDING_INTGRAD_FILES # New
    # The log for the last int_grad step is transferred with the final logs
)

for pid in "${PIDS_TO_WAIT[@]}"; do
    # Check if PID is non-empty before waiting
    if [ -n "$pid" ]; then
        if wait $pid; then
            echo "Background SCP/Process $pid completed successfully." | tee -a "$LOG_DIR/99_wait_downloads.log"
        else
            echo "ERROR: Background SCP/Process $pid failed with exit code $?." | tee -a "$LOG_DIR/99_wait_downloads.log"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "Warning: Skipping wait for an empty PID." | tee -a "$LOG_DIR/99_wait_downloads.log"
    fi
done

if [ $FAIL_COUNT -ne 0 ]; then
    echo "ERROR: $FAIL_COUNT background SCP/Process operation(s) failed. Check logs in $LOG_DIR (GCP) and $LOCAL_LOG_DEST (Mac) for details." | tee -a "$LOG_DIR/99_wait_downloads.log"
    # Decide if script should exit with error status
    # exit 1 # Uncomment this line if any SCP failure should make the whole script fail
fi

echo "All background downloads/processes attempted." | tee -a "$LOG_DIR/99_wait_downloads.log"

# Optional: Download remaining general logs and SCP logs
echo "Downloading remaining logs..." | tee "$LOG_DIR/99_final_logs.log"
set +e # Avoid exiting if final log scp fails
# Copy the last int_grad log file too
PID_SCP_LONG_EMBEDDING_INTGRAD_LOG_FINAL=$(start_log_scp "$LOG_DIR/24_intgrad_long_embedding.log" "$LOG_DIR/99_final_logs.log" "$LOCAL_LOG_DEST" "Longformer Embedding IntGrad Log")
# Copy all .log files from the LOG_DIR
SCP_FINAL_LOG_PID=$(scp -o BatchMode=yes -o StrictHostKeyChecking=no "$LOG_DIR"/*.log "$LOCAL_LOG_DEST" >> "$LOG_DIR/99_final_logs.log" 2>&1 & echo $!)

# Wait for the final log transfers
wait $PID_SCP_LONG_EMBEDDING_INTGRAD_LOG_FINAL || echo "Warning: Failed to SCP final Longformer Embedding IntGrad log." | tee -a "$LOG_DIR/99_final_logs.log"
wait $SCP_FINAL_LOG_PID || echo "Warning: Failed to SCP final logs." | tee -a "$LOG_DIR/99_final_logs.log"
set -e # Re-enable exit on error

echo "=================================================="
echo "Script finished."
date | tee "$LOG_DIR/99_end_time.log"
echo "=================================================="

# Send final completion log
scp -o BatchMode=yes -o StrictHostKeyChecking=no "$LOG_DIR/99_end_time.log" "$LOCAL_LOG_DEST" > /dev/null 2>&1 || true # Best effort

# Exit with 0 if no errors caused an early exit (due to set -e or explicit exit)
exit 0 