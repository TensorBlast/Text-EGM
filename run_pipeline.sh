#!/bin/bash

# Script to automate the Text-EGM pipeline on a GCP instance.
# It handles setup, preprocessing, training for 'big' and 'long' models,
# optional Tailscale setup for connectivity, and parallel downloads of
# artifacts and logs to a local Mac.

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
    echo "  - <local_base_dest_dir>/logs"
    echo "Also ensure passwordless SSH access from this GCP instance to the Mac is configured."
    exit 1
fi

if $USE_TAILSCALE && [ -z "$TAILSCALE_AUTH_KEY" ]; then
    echo "ERROR: --tailscale-auth-key is required when using --use-tailscale."
    exit 1
fi

# Construct destination paths for clarity
LOCAL_EMB_DEST="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/pretrained_embeddings/"
LOCAL_CKPT_DEST_BASE="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/runs/checkpoint/"
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
    if ! ( "$@" ) >> "$log_file" 2>&1; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> "$log_file"
        echo "ERROR: Command failed with exit code $?. See details above." >> "$log_file"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >> "$log_file"
        echo "ERROR: Command failed: $*" >&2 # Also print to stderr
        # Optionally exit script here if needed, but set -e handles this
        # exit 1
    fi
     echo "--------------------------------------------------" >> "$log_file"
     echo "Command finished: $*" >> "$log_file"
     echo "Timestamp: $(date)" >> "$log_file"
     echo "--------------------------------------------------" >> "$log_file"

}

# Starts an scp process in the background to copy files/dirs
start_scp() {
    local scp_log_file="$1" # Log file for this specific scp operation
    local source_path="$2"
    local dest_path="$3"
    local description="$4"

    echo "$(date): Starting background SCP: $description ($source_path -> $dest_path)" | tee -a "$scp_log_file"
    # Run scp in the background, redirecting its output to the specific log
    scp -r "$source_path" "$dest_path" >> "$scp_log_file" 2>&1 &
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
    scp "$main_log_file" "$dest_path" >> "$scp_log_file" 2>&1 &
    echo $! # Return the PID
}

# --- Main Script ---

echo "=================================================="
echo "Starting Full Text-EGM Pipeline on GCP Instance"
if $USE_TAILSCALE; then
    echo "Tailscale Mode: ENABLED"
fi
echo "Target Mac: ${MAC_USER}@${MAC_HOST}"
echo "Destination Base Directory: ${LOCAL_DEST_BASE_DIR}"
echo "=================================================="
mkdir -p "$LOG_DIR" # Ensure log directory exists early
date > "$LOG_DIR/00_start_time.log"

# 1. System Prep
echo "[Step 1/16] System Preparation..." | tee "$LOG_DIR/01_prep.log"
{
    echo "Ensuring wget and git are installed..."
    if ! command -v wget &> /dev/null; then
        echo "wget not found, attempting installation..."
        sudo apt-get update -y
        sudo apt-get install -y wget
    else
        echo "wget found."
    fi
     if ! command -v git &> /dev/null; then
        echo "git not found, attempting installation..."
        sudo apt-get update -y || true # Allow update to fail if already done
        sudo apt-get install -y git
    else
        echo "git found."
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
echo "[Step 2/16] Cloning Repository and Installing Requirements..." | tee "$LOG_DIR/02_setup.log"
if [ -d "$REPO_DIR" ]; then
    echo "Repository directory '$REPO_DIR' already exists. Skipping clone." | tee -a "$LOG_DIR/02_setup.log"
else
    log_exec "$LOG_DIR/02_setup.log" git clone "$REPO_URL"
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
    echo "[Step 3/16] Activating Tailscale..." | tee "$LOG_DIR/03_tailscale_up.log"
    echo "Attempting to bring Tailscale up..." >> "$LOG_DIR/03_tailscale_up.log"
    # Check current status first
    if sudo tailscale status > /dev/null 2>&1; then
        echo "Tailscale already seems to be running/configured. Status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
        sudo tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
        echo "Attempting 'tailscale up' anyway to ensure correct key/settings..." | tee -a "$LOG_DIR/03_tailscale_up.log"
    fi
    # Use --accept-routes to potentially access Mac services, adjust hostname
    # Using --ssh is optional but good practice if you might want to SSH later
    log_exec "$LOG_DIR/03_tailscale_up.log" sudo tailscale up --authkey="${TAILSCALE_AUTH_KEY}" --hostname="${GCP_HOSTNAME}" --accept-routes --ssh
    echo "Tailscale activation attempt finished. Final status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
    sudo tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
    echo "Sleeping for 5 seconds to allow Tailscale connection to establish..." | tee -a "$LOG_DIR/03_tailscale_up.log"
    sleep 5 # Give Tailscale a moment to connect/update routes
else
    echo "[Step 3/16] Skipping Tailscale Activation." | tee "$LOG_DIR/03_tailscale_up.log"
fi
# Note: Step numbering adjusted from here onwards

# 4. Preprocessing
echo "[Step 4/16] Running Preprocessing..." | tee "$LOG_DIR/04_preprocess.log"
log_exec "$LOG_DIR/04_preprocess.log" python preprocess/preprocess_intra.py
echo "Preprocessing finished."

# --- BigBird Model Pipeline ---
echo "--- Starting BigBird Pipeline ---"

# 5. Pretrain Embeddings (BigBird)
echo "[Step 5/16] Pretraining BigBird Embeddings..." | tee "$LOG_DIR/05_pretrain_big.log"
log_exec "$LOG_DIR/05_pretrain_big.log" python pretrain_embeddings.py --model big --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "BigBird Embedding Pretraining finished."

# 6. Download BigBird Embeddings (Background) + Log
echo "[Step 6/16] Downloading BigBird Embeddings and Log (background)..." | tee "$LOG_DIR/06_scp_big_emb.log"
PID_SCP_BIG_EMB_FILE=$(start_scp "$LOG_DIR/06_scp_big_emb.log" "./$PRETRAINED_EMB_DIR/big_embedding_weights.pt" "$LOCAL_EMB_DEST" "BigBird Embedding File")
PID_SCP_BIG_EMB_LOG=$(start_log_scp "$LOG_DIR/05_pretrain_big.log" "$LOG_DIR/06_scp_big_emb.log" "$LOCAL_LOG_DEST" "BigBird Pretrain Log")

# 7. Train BigBird (Default)
echo "[Step 7/16] Training BigBird (Default)..." | tee "$LOG_DIR/07_train_big_default.log"
# Run and capture output simultaneously to file and variable
# Need to handle potential errors within the Python script itself if `set -e` is too aggressive
set +e # Temporarily disable exit on error for Python script capture
TRAIN_BIG_DEFAULT_OUTPUT=$(python train.py --model big --epochs 50 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/07_train_big_default.log")
PYTHON_EXIT_CODE=$?
set -e # Re-enable exit on error
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/07_train_big_default.log"
    exit $PYTHON_EXIT_CODE
fi

BIG_DEFAULT_DIR_REL=$(echo "$TRAIN_BIG_DEFAULT_OUTPUT" | grep 'Directory created:' | sed 's/Directory created: \.\///' || echo "NOT_FOUND")

if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird default checkpoint directory from training output." | tee -a "$LOG_DIR/07_train_big_default.log"
    # Attempt to find the newest directory matching the pattern as a fallback
    BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | head -n 1 || echo "NOT_FOUND")
    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/07_train_big_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
    fi
fi
echo "BigBird Default Training finished. Checkpoint relative path: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"

# 8. Download BigBird Default Checkpoint (Background) + Log
echo "[Step 8/16] Downloading BigBird Default Checkpoint and Log (background)..." | tee "$LOG_DIR/08_scp_big_default_ckpt.log"
PID_SCP_BIG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/08_scp_big_default_ckpt.log" "$BIG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Default Checkpoint")
PID_SCP_BIG_DEFAULT_LOG=$(start_log_scp "$LOG_DIR/07_train_big_default.log" "$LOG_DIR/08_scp_big_default_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Default Train Log")

# 9. Train BigBird (Pretrained Embeddings)
echo "[Step 9/16] Training BigBird (Pretrained Embeddings)..." | tee "$LOG_DIR/09_train_big_embedding.log"
set +e
TRAIN_BIG_EMB_OUTPUT=$(python train.py --model big --epochs 50 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/big_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/09_train_big_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (BigBird Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/09_train_big_embedding.log"
    exit $PYTHON_EXIT_CODE
fi

BIG_EMBEDDING_DIR_REL=$(echo "$TRAIN_BIG_EMB_OUTPUT" | grep 'Directory created:' | sed 's/Directory created: \.\///' || echo "NOT_FOUND")

if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine BigBird embedding checkpoint directory from training output." | tee -a "$LOG_DIR/09_train_big_embedding.log"
    # Fallback
    BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/09_train_big_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
    fi
fi
echo "BigBird Pretrained Embedding Training finished. Checkpoint relative path: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"

# 10. Download BigBird Embedding Checkpoint (Background) + Log
echo "[Step 10/16] Downloading BigBird Embedding Checkpoint and Log (background)..." | tee "$LOG_DIR/10_scp_big_embedding_ckpt.log"
PID_SCP_BIG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/10_scp_big_embedding_ckpt.log" "$BIG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Embedding Checkpoint")
PID_SCP_BIG_EMBEDDING_LOG=$(start_log_scp "$LOG_DIR/09_train_big_embedding.log" "$LOG_DIR/10_scp_big_embedding_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Embedding Train Log")


# --- Longformer Model Pipeline ---
echo "--- Starting Longformer Pipeline ---"

# 11. Pretrain Embeddings (Longformer)
echo "[Step 11/16] Pretraining Longformer Embeddings..." | tee "$LOG_DIR/11_pretrain_long.log"
log_exec "$LOG_DIR/11_pretrain_long.log" python pretrain_embeddings.py --model long --epochs 30 --batch 16 --output_dir ./"$PRETRAINED_EMB_DIR"
echo "Longformer Embedding Pretraining finished."

# 12. Download Longformer Embeddings (Background) + Log
echo "[Step 12/16] Downloading Longformer Embeddings and Log (background)..." | tee "$LOG_DIR/12_scp_long_emb.log"
PID_SCP_LONG_EMB_FILE=$(start_scp "$LOG_DIR/12_scp_long_emb.log" "./$PRETRAINED_EMB_DIR/long_embedding_weights.pt" "$LOCAL_EMB_DEST" "Longformer Embedding File")
PID_SCP_LONG_EMB_LOG=$(start_log_scp "$LOG_DIR/11_pretrain_long.log" "$LOG_DIR/12_scp_long_emb.log" "$LOCAL_LOG_DEST" "Longformer Pretrain Log")

# 13. Train Longformer (Default)
echo "[Step 13/16] Training Longformer (Default)..." | tee "$LOG_DIR/13_train_long_default.log"
set +e
TRAIN_LONG_DEFAULT_OUTPUT=$(python train.py --model long --epochs 50 --batch 4 --no-use_ce 2>&1 | tee -a "$LOG_DIR/13_train_long_default.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/13_train_long_default.log"
    exit $PYTHON_EXIT_CODE
fi

LONG_DEFAULT_DIR_REL=$(echo "$TRAIN_LONG_DEFAULT_OUTPUT" | grep 'Directory created:' | sed 's/Directory created: \.\///' || echo "NOT_FOUND")

if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer default checkpoint directory from training output." | tee -a "$LOG_DIR/13_train_long_default.log"
    # Fallback - ensure we don't match the pretrained_emb directory name
    LONG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/13_train_long_default.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
    fi
fi
echo "Longformer Default Training finished. Checkpoint relative path: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"

# 14. Download Longformer Default Checkpoint (Background) + Log
echo "[Step 14/16] Downloading Longformer Default Checkpoint and Log (background)..." | tee "$LOG_DIR/14_scp_long_default_ckpt.log"
PID_SCP_LONG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/14_scp_long_default_ckpt.log" "$LONG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Default Checkpoint")
PID_SCP_LONG_DEFAULT_LOG=$(start_log_scp "$LOG_DIR/13_train_long_default.log" "$LOG_DIR/14_scp_long_default_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Default Train Log")

# 15. Train Longformer (Pretrained Embeddings)
echo "[Step 15/16] Training Longformer (Pretrained Embeddings)..." | tee "$LOG_DIR/15_train_long_embedding.log"
set +e
TRAIN_LONG_EMB_OUTPUT=$(python train.py --model long --epochs 50 --batch 4 --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/long_embedding_weights.pt 2>&1 | tee -a "$LOG_DIR/15_train_long_embedding.log")
PYTHON_EXIT_CODE=$?
set -e
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Python training script (Longformer Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/15_train_long_embedding.log"
    exit $PYTHON_EXIT_CODE
fi

LONG_EMBEDDING_DIR_REL=$(echo "$TRAIN_LONG_EMB_OUTPUT" | grep 'Directory created:' | sed 's/Directory created: \.\///' || echo "NOT_FOUND")

if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
    echo "ERROR: Could not determine Longformer embedding checkpoint directory from training output." | tee -a "$LOG_DIR/15_train_long_embedding.log"
    # Fallback
    LONG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
     if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
         echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/15_train_long_embedding.log"
         exit 1
    else
         echo "WARNING: Used fallback to find checkpoint directory: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/15_train_long_embedding.log"
    fi
fi
echo "Longformer Pretrained Embedding Training finished. Checkpoint relative path: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/15_train_long_embedding.log"

# 16. Download Longformer Embedding Checkpoint (Background) + Log
echo "[Step 16/16] Downloading Longformer Embedding Checkpoint and Log (background)..." | tee "$LOG_DIR/16_scp_long_embedding_ckpt.log"
PID_SCP_LONG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/16_scp_long_embedding_ckpt.log" "$LONG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Embedding Checkpoint")
PID_SCP_LONG_EMBEDDING_LOG=$(start_log_scp "$LOG_DIR/15_train_long_embedding.log" "$LOG_DIR/16_scp_long_embedding_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Embedding Train Log")


# --- Wait for all background downloads ---
echo "--- Waiting for background downloads to complete ---" | tee "$LOG_DIR/99_wait_downloads.log"
FAIL_COUNT=0
PIDS_TO_WAIT=(
    $PID_SCP_BIG_EMB_FILE
    $PID_SCP_BIG_EMB_LOG
    $PID_SCP_BIG_DEFAULT_CKPT
    $PID_SCP_BIG_DEFAULT_LOG
    $PID_SCP_BIG_EMBEDDING_CKPT
    $PID_SCP_BIG_EMBEDDING_LOG
    $PID_SCP_LONG_EMB_FILE
    $PID_SCP_LONG_EMB_LOG
    $PID_SCP_LONG_DEFAULT_CKPT
    $PID_SCP_LONG_DEFAULT_LOG
    $PID_SCP_LONG_EMBEDDING_CKPT
    $PID_SCP_LONG_EMBEDDING_LOG
)

for pid in "${PIDS_TO_WAIT[@]}"; do
    # Check if PID is non-empty before waiting
    if [ -n "$pid" ]; then
        if wait $pid; then
            echo "Background SCP process $pid completed successfully." | tee -a "$LOG_DIR/99_wait_downloads.log"
        else
            echo "ERROR: Background SCP process $pid failed with exit code $?." | tee -a "$LOG_DIR/99_wait_downloads.log"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "Warning: Skipping wait for an empty PID." | tee -a "$LOG_DIR/99_wait_downloads.log"
    fi
done

if [ $FAIL_COUNT -ne 0 ]; then
    echo "ERROR: $FAIL_COUNT SCP operation(s) failed. Check logs in $LOG_DIR (GCP) and $LOCAL_LOG_DEST (Mac) for details." | tee -a "$LOG_DIR/99_wait_downloads.log"
    # Decide if script should exit with error status
    # exit 1 # Uncomment this line if any SCP failure should make the whole script fail
fi

echo "All background downloads attempted." | tee -a "$LOG_DIR/99_wait_downloads.log"

# Optional: Download remaining general logs and SCP logs
echo "Downloading remaining logs..." | tee "$LOG_DIR/99_final_logs.log"
set +e # Avoid exiting if final log scp fails
SCP_FINAL_LOG_PID=$(scp -r "$LOG_DIR"/*.log "$LOCAL_LOG_DEST" >> "$LOG_DIR/99_final_logs.log" 2>&1 & echo $!)
wait $SCP_FINAL_LOG_PID || echo "Warning: Failed to SCP final logs." | tee -a "$LOG_DIR/99_final_logs.log"
set -e # Re-enable exit on error

echo "=================================================="
echo "Script finished."
date | tee "$LOG_DIR/99_end_time.log"
echo "=================================================="

# Send final completion log
scp "$LOG_DIR/99_end_time.log" "$LOCAL_LOG_DEST" > /dev/null 2>&1 || true # Best effort

# Exit with 0 if no SCP failures caused an early exit, or non-zero if they did (and exit 1 was uncommented)
exit 0 