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
SCRIPT_DIR="$(pwd)" # Save the directory where the script is executed
LOG_DIR="${SCRIPT_DIR}/run_logs" # Make LOG_DIR an absolute path
PRETRAINED_EMB_DIR="pretrained_embeddings"
CHECKPOINT_BASE_DIR="runs/checkpoint"
GCP_HOSTNAME="gcp-egm-runner" # Hostname for Tailscale

# Default batch sizes
TRAIN_BATCH_SIZE=8
INFER_BATCH_SIZE=16
RESUME_FROM=1  # Default to starting from the first step

# Detect if running in Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER=true
    echo "Docker environment detected."
fi

# --- Parameter Parsing ---
# Default values
USE_TAILSCALE=false
TAILSCALE_AUTH_KEY=""
MAC_USER=""
MAC_HOST=""
LOCAL_DEST_BASE_DIR=""
LOCAL_LOG_DEST=""
SCP_PORT="22"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --use-tailscale)
      USE_TAILSCALE=true
      shift
      ;;
    --tailscale-auth-key)
      TAILSCALE_AUTH_KEY="$2"
      shift 2
      ;;
    --user)
      MAC_USER="$2"
      shift 2
      ;;
    --host)
      MAC_HOST="$2"
      shift 2
      ;;
    --dest)
      LOCAL_DEST_BASE_DIR="$2"
      shift 2
      ;;
    --log-dest)
      LOCAL_LOG_DEST="$2"
      shift 2
      ;;
    --port)
      SCP_PORT="$2"
      shift 2
      ;;
    --train-batch)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --infer-batch)
      INFER_BATCH_SIZE="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    *)
      # Positional arguments (legacy support)
      if [ -z "$MAC_USER" ]; then
        MAC_USER="$1"
      elif [ -z "$MAC_HOST" ]; then
        MAC_HOST="$1"
      elif [ -z "$LOCAL_DEST_BASE_DIR" ]; then
        LOCAL_DEST_BASE_DIR="$1"
      else
        echo "Unknown option: $1"
        echo "Usage: $0 [--use-tailscale] [--tailscale-auth-key KEY] [--user USER] [--host HOST] [--dest DEST] [--log-dest LOG_DEST] [--port PORT] [--train-batch SIZE] [--infer-batch SIZE] [--resume-from STEP]"
        exit 1
      fi
      shift
      ;;
  esac
done

echo "Using training batch size: $TRAIN_BATCH_SIZE"
echo "Using inference batch size: $INFER_BATCH_SIZE"
echo "Resuming from step: $RESUME_FROM"

# Total number of steps in the pipeline - define this early
TOTAL_STEPS=24 # Update this if steps are added or removed

# --- Process Argument Values ---
# Set local log destination if not specified
if [ -z "$LOCAL_LOG_DEST" ]; then
    LOCAL_LOG_DEST="${LOCAL_DEST_BASE_DIR}/logs"
fi

# Echo configuration values
echo "SSH Config: ${MAC_USER}@${MAC_HOST}:${SCP_PORT} -> ${LOCAL_DEST_BASE_DIR}"
echo "Log destination: ${LOCAL_LOG_DEST}"

# --- Validate Arguments ---
if [ -z "$MAC_USER" ] || [ -z "$MAC_HOST" ] || [ -z "$LOCAL_DEST_BASE_DIR" ]; then
    echo "Usage: $0 [--use-tailscale] [--tailscale-auth-key KEY] --user USER --host HOST --dest DEST [--log-dest LOG_DEST] [--port PORT] [--train-batch SIZE] [--infer-batch SIZE] [--resume-from STEP]"
    echo "Example (no Tailscale): $0 myuser 192.168.1.100 /Users/myuser/gcp_results"
    echo "Example (with Tailscale): $0 --tailscale-auth-key tskey-... myuser my-mac-tailscale-name /Users/myuser/gcp_results"
    echo ""
    echo "ERROR: Missing required arguments (mac_user, mac_host_or_ip_or_ts_name, local_base_dest_dir)."
    echo "If using --tailscale-auth-key, --tailscale-auth-key is required, and <mac_host_or_ip_or_ts_name> should be the Mac's Tailscale name or IP."
    echo "Please ensure the following directories exist and are writable on the Mac at <local_base_dest_dir>:"
    echo "  - <local_base_dest_dir>/pretrained_embeddings"
    echo "  - <local_base_dest_dir>/runs/checkpoint"
    echo "  - <local_base_dest_dir>/int_grad_outputs" # New required directory
    echo "  - <local_base_dest_dir>/logs"
    echo "Also ensure passwordless SSH access (or Tailscale SSH) from this GCP instance to the Mac is configured."
    exit 1
fi

if [ -z "$TAILSCALE_AUTH_KEY" ] && [ "$USE_TAILSCALE" = true ]; then
    echo "ERROR: --tailscale-auth-key is required when using --use-tailscale."
    exit 1
fi

# Construct destination paths for clarity
LOCAL_EMB_DEST="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/pretrained_embeddings/"
LOCAL_CKPT_DEST_BASE="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/runs/checkpoint/"
LOCAL_INT_GRAD_DEST_BASE="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/int_grad_outputs/"
LOCAL_LOG_DEST="${MAC_USER}@${MAC_HOST}:${LOCAL_DEST_BASE_DIR}/logs/"

# --- Helper Functions ---

# Function to check if a step should be run based on RESUME_FROM
should_run_step() {
    local step_num="$1"
    if [ "$step_num" -ge "$RESUME_FROM" ]; then
        return 0  # True, should run
    else
        return 1  # False, should skip
    fi
}

# Function to skip a step with a message
skip_step() {
    local step_num="$1"
    local step_desc="$2"
    local log_file="$3"
    
    echo "[Step $step_num/$TOTAL_STEPS] Skipping: $step_desc (resuming from step $RESUME_FROM)" | tee "$log_file"
    return 0
}

# Display available steps for reference if requested
if [ "$RESUME_FROM" = "list" ]; then
    echo "Available steps in the pipeline:"
    echo "1: System Preparation"
    echo "2: Clone Repository and Install Requirements"
    echo "3: Optional: Activate Tailscale"
    echo "4: Preprocessing"
    echo "5: Pretrain BigBird Embeddings"
    echo "6: Download BigBird Embeddings (Background) + Log"
    echo "7: Train BigBird (Default)"
    echo "8: Download BigBird Default Checkpoint (Background) + Log"
    echo "9: Run Inference (BigBird Default)"
    echo "10: Run Integrated Gradients (BigBird Default)"
    echo "11: Train BigBird (Pretrained Embeddings)"
    echo "12: Download BigBird Embedding Checkpoint (Background) + Logs"
    echo "13: Run Inference (BigBird Embedding)" 
    echo "14: Run Integrated Gradients (BigBird Embedding)"
    echo "15: Pretrain Longformer Embeddings"
    echo "16: Download Longformer Embeddings (Background) + Logs"
    echo "17: Train Longformer (Default)"
    echo "18: Download Longformer Default Checkpoint (Background) + Log"
    echo "19: Run Inference (Longformer Default)"
    echo "20: Run Integrated Gradients (Longformer Default)"
    echo "21: Train Longformer (Pretrained Embeddings)"
    echo "22: Download Longformer Embedding Checkpoint (Background) + Logs"
    echo "23: Run Inference (Longformer Embedding)"
    echo "24: Run Integrated Gradients (Longformer Embedding)"
    exit 0
fi

# Validate RESUME_FROM is a number
if ! [[ "$RESUME_FROM" =~ ^[0-9]+$ ]]; then
    # Skip validation if RESUME_FROM is "list"
    if [ "$RESUME_FROM" != "list" ]; then
        echo "ERROR: --resume-from must be a number (1-24) or 'list'"
        echo "Use --resume-from list to see all available steps"
        exit 1
    fi
fi

# Validate RESUME_FROM is in range
if [[ "$RESUME_FROM" =~ ^[0-9]+$ ]] && ([ "$RESUME_FROM" -lt 1 ] || [ "$RESUME_FROM" -gt "$TOTAL_STEPS" ]); then
    echo "ERROR: --resume-from must be between 1 and $TOTAL_STEPS"
    echo "Use --resume-from list to see all available steps"
    exit 1
fi

# Function to delete .npy files in a given directory
delete_npy_files() {
    local target_dir="$1"
    local log_file="$2"
    
    echo "Deleting .npy files in directory: $target_dir" | tee -a "$log_file"
    
    # Check if the directory exists
    if [ ! -d "$target_dir" ]; then
        echo "WARNING: Directory $target_dir does not exist. Skipping .npy file deletion." | tee -a "$log_file"
        return 0
    fi
    
    # Count .npy files before deletion
    local npy_count=$(find "$target_dir" -maxdepth 1 -name "*.npy" -type f | wc -l || echo "0")
    echo "Found $npy_count .npy files in $target_dir" | tee -a "$log_file"
    
    # Delete .npy files, redirecting stderr to suppress any error messages
    if [ "$npy_count" -gt 0 ]; then
        find "$target_dir" -maxdepth 1 -name "*.npy" -type f -exec rm -f {} \; 2>/dev/null || {
            echo "WARNING: Some errors occurred during deletion. Some files may not have been deleted." | tee -a "$log_file"
        }
        echo ".npy files deletion attempted" | tee -a "$log_file"
        
        # Verify deletion
        local remaining=$(find "$target_dir" -maxdepth 1 -name "*.npy" -type f | wc -l || echo "unknown")
        if [ "$remaining" = "0" ]; then
            echo "All .npy files successfully deleted from $target_dir" | tee -a "$log_file"
        else
            echo "WARNING: $remaining .npy files still remain in $target_dir" | tee -a "$log_file"
        fi
    else
        echo "No .npy files found in $target_dir. Nothing to delete." | tee -a "$log_file"
    fi
    
    echo ".npy files deletion operation complete" | tee -a "$log_file"
    return 0  # Always return success to prevent script termination
}

# Executes a command, logging stdout and stderr to a file and displaying to console
log_exec() {
    local log_file="$1"
    shift
    echo "--------------------------------------------------" | tee -a "$log_file"
    echo "Running command: $*" | tee -a "$log_file"
    echo "Timestamp: $(date)" | tee -a "$log_file"
    echo "--------------------------------------------------" | tee -a "$log_file"
    
    # Execute command, tee to log file and display to console
    local cmd="$*"
    local use_progress_indicator=false
    
    # Check for commands that might need a progress indicator
    if [[ "$cmd" == *"int_grad.py"* ]] || [[ "$cmd" == *"inference.py"* ]] || [[ "$cmd" == *"train.py"* ]]; then
        use_progress_indicator=true
        echo "Running long process with progress updates..." | tee -a "$log_file"
        echo "This may take a while - progress indicator will run every 30 seconds if no output is seen." | tee -a "$log_file"
    fi
    
    # For Python commands, ensure unbuffered output
    if [[ "$cmd" == python* ]]; then
        # Set PYTHONUNBUFFERED for Python commands to prevent output buffering
        cmd="PYTHONUNBUFFERED=1 stdbuf -oL -eL $cmd"
    fi
    
    # Start progress indicator if needed
    if [ "$use_progress_indicator" = true ]; then
        (
            i=0
            while true; do
                sleep 30
                i=$((i+30))
                echo "[Progress indicator] Still running... ($i seconds elapsed)" | tee -a "$log_file"
            done
        ) &
        PROGRESS_PID=$!
        
        # Make sure to kill the progress indicator when we're done
        trap "kill $PROGRESS_PID 2>/dev/null || true" EXIT
    fi
    
    # Use script to run the command and capture ALL output, including carriage returns
    # This is essential for properly displaying progress bars
    if command -v script >/dev/null 2>&1; then
        # Use script to capture all output including carriage returns
        # The "cat" ensures continuous output
        script -q -c "$cmd" /dev/null | tee -a "$log_file" || {
            local exit_code=$?
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
            echo "ERROR: Command failed with exit code $exit_code." | tee -a "$log_file"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
            echo "ERROR: Command failed: $*" >&2 # Also print to stderr
            
            # Kill progress indicator if it exists
            if [ "$use_progress_indicator" = true ]; then
                kill $PROGRESS_PID 2>/dev/null || true
            fi
            
            return $exit_code
        }
    else
        # Fallback if script is not available
        eval "$cmd" 2>&1 | tee -a "$log_file" || {
            local exit_code=$?
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
            echo "ERROR: Command failed with exit code $exit_code." | tee -a "$log_file"
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$log_file"
            echo "ERROR: Command failed: $*" >&2 # Also print to stderr
            
            # Kill progress indicator if it exists
            if [ "$use_progress_indicator" = true ]; then
                kill $PROGRESS_PID 2>/dev/null || true
            fi
            
            return $exit_code
        }
    fi
    
    # Kill progress indicator if it exists
    if [ "$use_progress_indicator" = true ]; then
        kill $PROGRESS_PID 2>/dev/null || true
    fi
    
    echo "--------------------------------------------------" | tee -a "$log_file"
    echo "Command finished: $*" | tee -a "$log_file"
    echo "Timestamp: $(date)" | tee -a "$log_file"
    echo "--------------------------------------------------" | tee -a "$log_file"
    # Ensure the function returns success if the command succeeded
    return 0
}

# Starts an scp process in the background to copy files/dirs
start_scp() {
    local scp_log_file="$1" # Log file for this specific scp operation
    local source_path="$2"
    local dest_path="$3"
    local description="$4"

    # Normalize paths to avoid double slashes
    source_path=$(echo "$source_path" | sed 's|//|/|g')
    dest_path=$(echo "$dest_path" | sed 's|//|/|g')

    echo "$(date): Starting background SCP: $description ($source_path -> $dest_path)" | tee -a "$scp_log_file"
    
    # Check if source exists
    if [ ! -e "$source_path" ]; then
        echo "WARNING: Source path $source_path does not exist. Cannot perform SCP." | tee -a "$scp_log_file"
        return 0  # Return a dummy PID
    fi
    
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

    # Normalize paths to avoid double slashes
    main_log_file=$(echo "$main_log_file" | sed 's|//|/|g')
    dest_path=$(echo "$dest_path" | sed 's|//|/|g')

    echo "$(date): Starting background SCP for log: $description ($main_log_file -> $dest_path)" | tee -a "$scp_log_file"
    
    # Check if log file exists before copying
    if [ ! -f "$main_log_file" ]; then
        echo "WARNING: Log file $main_log_file does not exist. Creating empty log file before copying." | tee -a "$scp_log_file"
        touch "$main_log_file" # Create empty file to avoid errors
    fi
    
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
    
    # Normalize paths to avoid double slashes
    source_checkpoint_dir=$(echo "$source_checkpoint_dir" | sed 's|//|/|g')
    dest_base_path=$(echo "$dest_base_path" | sed 's|//|/|g')
    
    local ckpt_basename=$(basename "$source_checkpoint_dir")
    local final_dest_path="${dest_base_path}${ckpt_basename}/" # Append the specific dir name

    echo "$(date): Starting background SCP for IntGrad results: $description ($source_checkpoint_dir/*.png, *.npy -> $final_dest_path)" | tee -a "$scp_log_file"

    # Check if source directory exists
    if [ ! -d "$source_checkpoint_dir" ]; then
        echo "WARNING: Source checkpoint directory $source_checkpoint_dir does not exist. Cannot copy IntGrad results." | tee -a "$scp_log_file"
        return 0
    fi
    
    # Check if there are any PNG or NPY files to copy
    local file_count=$(find "$source_checkpoint_dir" -maxdepth 1 \( -name '*.png' -o -name '*.npy' \) | wc -l)
    if [ "$file_count" -eq 0 ]; then
        echo "WARNING: No PNG or NPY files found in $source_checkpoint_dir. Nothing to copy." | tee -a "$scp_log_file"
        return 0
    else
        echo "Found $file_count PNG/NPY files to copy from $source_checkpoint_dir" | tee -a "$scp_log_file"
    fi

    # Create the destination directory on the Mac first
    # Use ssh to run mkdir -p remotely. Ignore errors if it already exists.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${MAC_USER}@${MAC_HOST}" "mkdir -p '${LOCAL_DEST_BASE_DIR}/int_grad_outputs/${ckpt_basename}'" >> "$scp_log_file" 2>&1 || echo "Warning: mkdir -p on remote might have failed (possibly dir exists)" >> "$scp_log_file"

    # Use find to copy only specific file types, avoids copying the large checkpoint file again
    find "$source_checkpoint_dir" -maxdepth 1 \( -name '*.png' -o -name '*.npy' \) -exec \
        scp -o BatchMode=yes -o StrictHostKeyChecking=no {} "$final_dest_path" >> "$scp_log_file" 2>&1 \; &

    echo $! # Return the PID of the background find/scp process
}

# Function to sanitize paths by removing problematic characters
sanitize_path() {
    local path="$1"
    # Remove carriage returns, tabs, and other control characters
    # First, normalize double slashes
    path=$(echo "$path" | sed 's|//|/|g')
    # Then remove carriage returns and other control characters
    path=$(echo "$path" | tr -d '\r\n\t')
    # Clean up any trailing or leading whitespace
    path=$(echo "$path" | xargs)
    echo "$path"
}

# --- Main Script ---

echo "=================================================="
echo "Starting Full Text-EGM Pipeline on GCP Instance"
if [ "$USE_TAILSCALE" = true ]; then
    echo "Tailscale Mode: ENABLED"
fi
echo "Target Mac: ${MAC_USER}@${MAC_HOST}"
echo "Destination Base Directory: ${LOCAL_DEST_BASE_DIR}"
echo "=================================================="

# Create log directory early with absolute path
mkdir -p "$LOG_DIR" # Ensure log directory exists early
chmod 755 "$LOG_DIR" # Ensure proper permissions

# Log the directories being used
echo "Using log directory: $LOG_DIR"
echo "Using script directory: $SCRIPT_DIR"

date > "$LOG_DIR/00_start_time.log"

# 1. System Prep
if should_run_step 1; then
    echo "[Step 1/$TOTAL_STEPS] System Preparation..." | tee "$LOG_DIR/01_prep.log"
    {
        echo "Ensuring wget, git, and findutils are installed..."
        if ! command -v wget &> /dev/null || ! command -v git &> /dev/null || ! command -v find &> /dev/null; then
            echo "One or more required tools (wget, git, find) not found, attempting installation..."
            # Assuming Debian/Ubuntu based system
             apt-get update -y
             apt-get install -y wget git findutils
        else
            echo "wget, git, find found."
        fi

        if [ "$USE_TAILSCALE" = true ]; then
            if $IN_DOCKER; then
                echo "WARNING: Tailscale requested, but we're in a Docker container."
                echo "Tailscale requires special setup in Docker. Consider these options:"
                echo "1. Run the container with --network=host --privileged"
                echo "2. Use host networking instead of Tailscale"
                echo "3. Mount the Tailscale socket from host: -v /var/run/tailscale:/var/run/tailscale"
                echo "For now, continuing without Tailscale..."
                USE_TAILSCALE=false
            else
                echo "Ensuring Tailscale is installed..."
                if ! command -v tailscale &> /dev/null; then
                    echo "Tailscale not found, installing..."
                    curl -fsSL https://tailscale.com/install.sh | sh
                else
                    echo "Tailscale found."
                fi
            fi
        fi
        echo "System preparation checks complete."
    } 2>&1 | tee -a "$LOG_DIR/01_prep.log"
    echo "System preparation finished."
else
    skip_step 1 "System Preparation" "$LOG_DIR/01_prep.log"
fi

# 2. Clone Repo and Setup
if should_run_step 2; then
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
else
    skip_step 2 "Cloning Repository and Installing Requirements" "$LOG_DIR/02_setup.log"
    # Even if we skip this step, we need to ensure we're in the right directory
    if [ -d "$REPO_DIR" ]; then
        cd "$REPO_DIR"
        echo "Changed to repository directory: $(pwd)" | tee -a "$LOG_DIR/02_setup.log"
    else
        echo "ERROR: Repository directory '$REPO_DIR' does not exist and step 2 was skipped." | tee -a "$LOG_DIR/02_setup.log"
        echo "Cannot continue without repository. Please run without --resume-from option first." | tee -a "$LOG_DIR/02_setup.log"
        exit 1
    fi
fi

# 3. Optional: Activate Tailscale
if should_run_step 3; then
    if [ "$USE_TAILSCALE" = true ]; then
        echo "[Step 3/$TOTAL_STEPS] Activating Tailscale..." | tee "$LOG_DIR/03_tailscale_up.log"
        echo "Attempting to bring Tailscale up..." >> "$LOG_DIR/03_tailscale_up.log"
        # Check current status first
        if tailscale status > /dev/null 2>&1; then
            echo "Tailscale already seems to be running/configured. Status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
            tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
            echo "Attempting 'tailscale up' anyway to ensure correct key/settings... Use 'tailscale logout' to reset if needed." | tee -a "$LOG_DIR/03_tailscale_up.log"
        fi
        # Use --accept-routes to potentially access Mac services, adjust hostname
        # Using --ssh enables Tailscale SSH server on the GCP instance (optional but useful)
        # Use quotes around the auth key in case it contains special characters
        log_exec "$LOG_DIR/03_tailscale_up.log" tailscale up --authkey="$TAILSCALE_AUTH_KEY" --hostname="$GCP_HOSTNAME" --accept-routes --ssh
        echo "Tailscale activation attempt finished. Final status:" | tee -a "$LOG_DIR/03_tailscale_up.log"
        tailscale status | tee -a "$LOG_DIR/03_tailscale_up.log"
        echo "Sleeping for 5 seconds to allow Tailscale connection to establish..." | tee -a "$LOG_DIR/03_tailscale_up.log"
        sleep 5 # Give Tailscale a moment to connect/update routes
    else
        echo "[Step 3/$TOTAL_STEPS] Skipping Tailscale Activation." | tee "$LOG_DIR/03_tailscale_up.log"
        if $IN_DOCKER; then
            echo "Running in Docker container without Tailscale." | tee -a "$LOG_DIR/03_tailscale_up.log"
            echo "Make sure your container has proper network connectivity to the Mac." | tee -a "$LOG_DIR/03_tailscale_up.log"
        fi
    fi
else
    skip_step 3 "Tailscale Activation" "$LOG_DIR/03_tailscale_up.log"
fi

# 4. Preprocessing
if should_run_step 4; then
    echo "[Step 4/$TOTAL_STEPS] Running Preprocessing..." | tee "$LOG_DIR/04_preprocess.log"
    # Check if preprocessed data files already exist
    DATA_DIR="../data"
    if [ -f "${DATA_DIR}/train_intra.npy" ] && [ -f "${DATA_DIR}/val_intra.npy" ] && [ -f "${DATA_DIR}/test_intra.npy" ]; then
        echo "Preprocessed data files already exist in ${DATA_DIR}. Skipping preprocessing." | tee -a "$LOG_DIR/04_preprocess.log"
    else
        echo "Preprocessed data files not found. Running preprocessing..." | tee -a "$LOG_DIR/04_preprocess.log"
        log_exec "$LOG_DIR/04_preprocess.log" python preprocess/preprocess_intra.py
        echo "Preprocessing finished."
    fi
else
    skip_step 4 "Preprocessing" "$LOG_DIR/04_preprocess.log"
fi

# --- BigBird Model Pipeline ---
echo "--- Starting BigBird Pipeline ---"

# 5. Pretrain Embeddings (BigBird)
if should_run_step 5; then
    echo "[Step 5/$TOTAL_STEPS] Pretraining BigBird Embeddings..." | tee "$LOG_DIR/05_pretrain_big.log"
    BIG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/big_embedding_weights.pt"
    if [ -f "$BIG_EMBEDDING_FILE" ]; then
        echo "BigBird embedding file already exists at $BIG_EMBEDDING_FILE. Skipping pretraining." | tee -a "$LOG_DIR/05_pretrain_big.log"
    else
        log_exec "$LOG_DIR/05_pretrain_big.log" python pretrain_embeddings.py --model big --epochs 30 --batch $INFER_BATCH_SIZE --output_dir ./"$PRETRAINED_EMB_DIR"
        echo "BigBird Embedding Pretraining finished."
    fi
else
    skip_step 5 "Pretraining BigBird Embeddings" "$LOG_DIR/05_pretrain_big.log"
    # Check if embedding file exists when skipping step
    BIG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/big_embedding_weights.pt"
    if [ ! -f "$BIG_EMBEDDING_FILE" ] && [ "$RESUME_FROM" -gt 5 ]; then
        echo "WARNING: BigBird embedding file does not exist at $BIG_EMBEDDING_FILE." | tee -a "$LOG_DIR/05_pretrain_big.log"
        echo "Some later steps may fail if they depend on this file." | tee -a "$LOG_DIR/05_pretrain_big.log"
    fi
fi

# 6. Download BigBird Embeddings (Background) + Log
if should_run_step 6; then
    echo "[Step 6/$TOTAL_STEPS] Downloading BigBird Embeddings and Log (background)..." | tee "$LOG_DIR/06_scp_big_emb.log"
    PID_SCP_BIG_EMB_FILE=$(start_scp "$LOG_DIR/06_scp_big_emb.log" "./$PRETRAINED_EMB_DIR/big_embedding_weights.pt" "$LOCAL_EMB_DEST" "BigBird Embedding File")
    PID_SCP_BIG_EMB_LOG=$(start_log_scp "$LOG_DIR/05_pretrain_big.log" "$LOG_DIR/06_scp_big_emb.log" "$LOCAL_LOG_DEST" "BigBird Pretrain Log")
else
    skip_step 6 "Downloading BigBird Embeddings and Log" "$LOG_DIR/06_scp_big_emb.log"
    # If skipping, initialize variables to prevent errors in wait section at the end
    PID_SCP_BIG_EMB_FILE=""
    PID_SCP_BIG_EMB_LOG=""
fi

# 7. Train BigBird (Default)
if should_run_step 7; then
    echo "[Step 7/$TOTAL_STEPS] Training BigBird (Default)..." | tee "$LOG_DIR/07_train_big_default.log"
    log_exec "$LOG_DIR/07_train_big_default.log" python train.py --model big --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/07_train_big_default.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    BIG_DEFAULT_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/07_train_big_default.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")

    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not determine BigBird default checkpoint directory from training output." | tee -a "$LOG_DIR/07_train_big_default.log"
        # Fallback is still useful if script output format changes unexpectedly
        BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
        if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/07_train_big_default.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
             # Sanitize the fallback path too
             BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")
        fi
    fi

    # Verify the directory exists
    if [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
        # Try a more aggressive fallback
        BIG_DEFAULT_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*big*" | grep -v 'pretrained_emb' | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ] || [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
            echo "ERROR: Could not find any BigBird default checkpoint directory." | tee -a "$LOG_DIR/07_train_big_default.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
        fi
    fi
    
    BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_DEFAULT_CKPT_NAME=$(sanitize_path "$BIG_DEFAULT_CKPT_NAME")
    
    echo "BigBird Default Training finished. Checkpoint directory: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/07_train_big_default.log"

    # Delete .npy files after training
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/07_train_big_default.log"
else
    skip_step 7 "Training BigBird (Default)" "$LOG_DIR/07_train_big_default.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not find BigBird default checkpoint directory from previous runs." | tee -a "$LOG_DIR/07_train_big_default.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/07_train_big_default.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/07_train_big_default.log"
        exit 1
    fi
    
    BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_DEFAULT_CKPT_NAME=$(sanitize_path "$BIG_DEFAULT_CKPT_NAME")
    
    echo "Found BigBird Default checkpoint from previous run: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/07_train_big_default.log"
fi

# 8. Download BigBird Default Checkpoint (Background) + Log
if should_run_step 8; then
    echo "[Step 8/$TOTAL_STEPS] Downloading BigBird Default Checkpoint and Log (background)..." | tee "$LOG_DIR/08_scp_big_default_ckpt.log"
    PID_SCP_BIG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/08_scp_big_default_ckpt.log" "$BIG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Default Checkpoint")
    PID_SCP_BIG_DEFAULT_LOG=$(start_log_scp "$LOG_DIR/07_train_big_default.log" "$LOG_DIR/08_scp_big_default_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Default Train Log")
else
    skip_step 8 "Downloading BigBird Default Checkpoint and Log" "$LOG_DIR/08_scp_big_default_ckpt.log"
    # If skipping, initialize variables to prevent errors in wait section at the end
    PID_SCP_BIG_DEFAULT_CKPT=""
    PID_SCP_BIG_DEFAULT_LOG=""
fi

# 9. Run Inference (BigBird Default)
if should_run_step 9; then
    echo "[Step 9/$TOTAL_STEPS] Running Inference (BigBird Default)..." | tee "$LOG_DIR/09_infer_big_default.log"
    log_exec "$LOG_DIR/09_infer_big_default.log" python inference.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --batch $INFER_BATCH_SIZE --mask 0.75
    PID_SCP_BIG_DEFAULT_INFER_LOG=$(start_log_scp "$LOG_DIR/09_infer_big_default.log" "$LOG_DIR/10_intgrad_big_default.log" "$LOCAL_LOG_DEST" "BigBird Default Inference Log")
    echo "Inference (BigBird Default) finished."

    # Delete .npy files after inference
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/09_infer_big_default.log"
else
    skip_step 9 "Running Inference (BigBird Default)" "$LOG_DIR/09_infer_big_default.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_BIG_DEFAULT_INFER_LOG=""
fi

# 10. Run Integrated Gradients (BigBird Default) + Start Downloads
if should_run_step 10; then
    echo "[Step 10/$TOTAL_STEPS] Running Integrated Gradients (BigBird Default)..." | tee "$LOG_DIR/10_intgrad_big_default.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/10_intgrad_big_default.log"
    log_exec "$LOG_DIR/10_intgrad_big_default.log" python visualize/int_grad.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/10_intgrad_big_default.log"
    echo "Integrated Gradients (BigBird Default) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/10_intgrad_big_default.log"

    # Start Downloads for IntGrad and previous Inference log
    echo "[Step 10/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (BigBird Default) (background)..." | tee -a "$LOG_DIR/10_intgrad_big_default.log"
    PID_SCP_BIG_DEFAULT_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/10_intgrad_big_default.log" "$BIG_DEFAULT_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "BigBird Default IntGrad Results")
    PID_SCP_BIG_DEFAULT_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/09_infer_big_default.log" "$LOG_DIR/10_intgrad_big_default.log" "$LOCAL_LOG_DEST" "BigBird Default IntGrad Log")
else
    skip_step 10 "Running Integrated Gradients (BigBird Default)" "$LOG_DIR/10_intgrad_big_default.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_BIG_DEFAULT_INTGRAD_FILES=""
    PID_SCP_BIG_DEFAULT_INTGRAD_LOG=""
fi

# 11. Train BigBird (Pretrained Embeddings)
if should_run_step 11; then
    echo "[Step 11/$TOTAL_STEPS] Training BigBird (Pretrained Embeddings)..." | tee "$LOG_DIR/11_train_big_embedding.log"
    log_exec "$LOG_DIR/11_train_big_embedding.log" python train.py --model big --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/big_embedding_weights.pt
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/11_train_big_embedding.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    BIG_EMBEDDING_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/11_train_big_embedding.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")

    if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not determine BigBird embedding checkpoint directory from training output." | tee -a "$LOG_DIR/11_train_big_embedding.log"
        # Fallback is still useful if script output format changes unexpectedly
        BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/11_train_big_embedding.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/11_train_big_embedding.log"
             # Sanitize the fallback path too
             BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")
        fi
    fi
    
    # Verify the directory exists
    if [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/11_train_big_embedding.log"
        # Try a more aggressive fallback
        BIG_EMBEDDING_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*big*pretrained_emb*" | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ] || [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
            echo "ERROR: Could not find any BigBird pretrained embedding checkpoint directory." | tee -a "$LOG_DIR/11_train_big_embedding.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/11_train_big_embedding.log"
        fi
    fi
    
    BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_EMBEDDING_CKPT_NAME=$(sanitize_path "$BIG_EMBEDDING_CKPT_NAME")
    
    echo "BigBird Pretrained Embedding Training finished. Checkpoint directory: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/11_train_big_embedding.log"

    # Delete .npy files after training
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/11_train_big_embedding.log"
else
    skip_step 11 "Training BigBird (Pretrained Embeddings)" "$LOG_DIR/11_train_big_embedding.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
    if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not find BigBird embedding checkpoint directory from previous runs." | tee -a "$LOG_DIR/11_train_big_embedding.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/11_train_big_embedding.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/11_train_big_embedding.log"
        exit 1
    fi
    
    BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_EMBEDDING_CKPT_NAME=$(sanitize_path "$BIG_EMBEDDING_CKPT_NAME")
    
    echo "Found BigBird Embedding checkpoint from previous run: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/11_train_big_embedding.log"
fi

# 12. Download BigBird Embedding Checkpoint (Background) + Logs
if should_run_step 12; then
    echo "[Step 12/$TOTAL_STEPS] Downloading BigBird Embedding Checkpoint and Logs (background)..." | tee "$LOG_DIR/12_scp_big_embedding_ckpt.log"
    PID_SCP_BIG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$BIG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "BigBird Embedding Checkpoint")
    PID_SCP_BIG_EMBEDDING_TRAIN_LOG=$(start_log_scp "$LOG_DIR/11_train_big_embedding.log" "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Embedding Train Log")
    PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/10_intgrad_big_default.log" "$LOG_DIR/12_scp_big_embedding_ckpt.log" "$LOCAL_LOG_DEST" "BigBird Default IntGrad Log")
else
    skip_step 12 "Downloading BigBird Embedding Checkpoint and Logs" "$LOG_DIR/12_scp_big_embedding_ckpt.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_BIG_EMBEDDING_CKPT=""
    PID_SCP_BIG_EMBEDDING_TRAIN_LOG=""
    PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=""
fi

# 13. Run Inference (BigBird Embedding)
if should_run_step 13; then
    echo "[Step 13/$TOTAL_STEPS] Running Inference (BigBird Embedding)..." | tee "$LOG_DIR/13_infer_big_embedding.log"
    log_exec "$LOG_DIR/13_infer_big_embedding.log" python inference.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --batch $INFER_BATCH_SIZE --mask 0.75
    PID_SCP_BIG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/13_infer_big_embedding.log" "$LOG_DIR/14_intgrad_big_embedding.log" "$LOCAL_LOG_DEST" "BigBird Embedding Inference Log")
    echo "Inference (BigBird Embedding) finished."

    # Delete .npy files after inference
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/13_infer_big_embedding.log"
else
    skip_step 13 "Running Inference (BigBird Embedding)" "$LOG_DIR/13_infer_big_embedding.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_BIG_EMBEDDING_INFER_LOG=""
fi

# 14. Run Integrated Gradients (BigBird Embedding) + Start Downloads
if should_run_step 14; then
    echo "[Step 14/$TOTAL_STEPS] Running Integrated Gradients (BigBird Embedding)..." | tee "$LOG_DIR/14_intgrad_big_embedding.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/14_intgrad_big_embedding.log"
    log_exec "$LOG_DIR/14_intgrad_big_embedding.log" python visualize/int_grad.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/14_intgrad_big_embedding.log"
    echo "Integrated Gradients (BigBird Embedding) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/14_intgrad_big_embedding.log"

    # Start Downloads for IntGrad and previous Inference log
    echo "[Step 14/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (BigBird Embedding) (background)..." | tee -a "$LOG_DIR/14_intgrad_big_embedding.log"
    PID_SCP_BIG_EMBEDDING_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/14_intgrad_big_embedding.log" "$BIG_EMBEDDING_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "BigBird Embedding IntGrad Results")
    PID_SCP_BIG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/13_infer_big_embedding.log" "$LOG_DIR/14_intgrad_big_embedding.log" "$LOCAL_LOG_DEST" "BigBird Embedding Inference Log")
else
    skip_step 14 "Running Integrated Gradients (BigBird Embedding)" "$LOG_DIR/14_intgrad_big_embedding.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_BIG_EMBEDDING_INTGRAD_FILES=""
    PID_SCP_BIG_EMBEDDING_INFER_LOG=""
fi

# --- Longformer Model Pipeline ---
echo "--- Starting Longformer Pipeline ---"

# 15. Pretrain Embeddings (Longformer)
if should_run_step 15; then
    echo "[Step 15/$TOTAL_STEPS] Pretraining Longformer Embeddings..." | tee "$LOG_DIR/15_pretrain_long.log"
    LONG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/long_embedding_weights.pt"
    if [ -f "$LONG_EMBEDDING_FILE" ]; then
        echo "Longformer embedding file already exists at $LONG_EMBEDDING_FILE. Skipping pretraining." | tee -a "$LOG_DIR/15_pretrain_long.log"
    else
        log_exec "$LOG_DIR/15_pretrain_long.log" python pretrain_embeddings.py --model long --epochs 30 --batch $INFER_BATCH_SIZE --output_dir ./"$PRETRAINED_EMB_DIR"
        echo "Longformer Embedding Pretraining finished."
    fi
else
    skip_step 15 "Pretraining Longformer Embeddings" "$LOG_DIR/15_pretrain_long.log"
    # Check if embedding file exists when skipping step
    LONG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/long_embedding_weights.pt"
    if [ ! -f "$LONG_EMBEDDING_FILE" ] && [ "$RESUME_FROM" -gt 15 ]; then
        echo "WARNING: Longformer embedding file does not exist at $LONG_EMBEDDING_FILE." | tee -a "$LOG_DIR/15_pretrain_long.log"
        echo "Some later steps may fail if they depend on this file." | tee -a "$LOG_DIR/15_pretrain_long.log"
    fi
fi

# 16. Download Longformer Embeddings (Background) + Logs
if should_run_step 16; then
    echo "[Step 16/$TOTAL_STEPS] Downloading Longformer Embeddings and Logs (background)..." | tee "$LOG_DIR/16_scp_long_emb.log"
    PID_SCP_LONG_EMB_FILE=$(start_scp "$LOG_DIR/16_scp_long_emb.log" "./$PRETRAINED_EMB_DIR/long_embedding_weights.pt" "$LOCAL_EMB_DEST" "Longformer Embedding File")
    PID_SCP_LONG_PRETRAIN_LOG=$(start_log_scp "$LOG_DIR/15_pretrain_long.log" "$LOG_DIR/16_scp_long_emb.log" "$LOCAL_LOG_DEST" "Longformer Pretrain Log")
    PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/14_intgrad_big_embedding.log" "$LOG_DIR/16_scp_long_emb.log" "$LOCAL_LOG_DEST" "BigBird Embedding IntGrad Log") # Previous step's log
else
    skip_step 16 "Downloading Longformer Embeddings and Logs" "$LOG_DIR/16_scp_long_emb.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_EMB_FILE=""
    PID_SCP_LONG_PRETRAIN_LOG=""
    PID_SCP_BIG_EMBEDDING_INTGRAD_LOG=""
fi

# 17. Train Longformer (Default)
if should_run_step 17; then
    echo "[Step 17/$TOTAL_STEPS] Training Longformer (Default)..." | tee "$LOG_DIR/17_train_long_default.log"
    log_exec "$LOG_DIR/17_train_long_default.log" python train.py --model long --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/17_train_long_default.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    LONG_DEFAULT_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/17_train_long_default.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    LONG_DEFAULT_DIR_REL=$(sanitize_path "$LONG_DEFAULT_DIR_REL")

    if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not determine Longformer default checkpoint directory from training output." | tee -a "$LOG_DIR/17_train_long_default.log"
        # Fallback is still useful if script output format changes unexpectedly
        LONG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
        if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/17_train_long_default.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/17_train_long_default.log"
             # Sanitize the fallback path too
             LONG_DEFAULT_DIR_REL=$(sanitize_path "$LONG_DEFAULT_DIR_REL")
        fi
    fi

    # Verify the directory exists
    if [ ! -d "$LONG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/17_train_long_default.log"
        # Try a more aggressive fallback
        LONG_DEFAULT_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*long*" | grep -v 'pretrained_emb' | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ] || [ ! -d "$LONG_DEFAULT_DIR_REL" ]; then
            echo "ERROR: Could not find any Longformer default checkpoint directory." | tee -a "$LOG_DIR/17_train_long_default.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/17_train_long_default.log"
        fi
    fi
    
    LONG_DEFAULT_CKPT_NAME=$(basename "$LONG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    LONG_DEFAULT_CKPT_NAME=$(sanitize_path "$LONG_DEFAULT_CKPT_NAME")
    
    echo "Longformer Default Training finished. Checkpoint directory: $LONG_DEFAULT_DIR_REL (Name: $LONG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/17_train_long_default.log"

    # Delete .npy files after training
    delete_npy_files "$LONG_DEFAULT_DIR_REL" "$LOG_DIR/17_train_long_default.log"
else
    skip_step 17 "Training Longformer (Default)" "$LOG_DIR/17_train_long_default.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    LONG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
    if [ "$LONG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not find Longformer default checkpoint directory from previous runs." | tee -a "$LOG_DIR/17_train_long_default.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/17_train_long_default.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    LONG_DEFAULT_DIR_REL=$(sanitize_path "$LONG_DEFAULT_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$LONG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $LONG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/17_train_long_default.log"
        exit 1
    fi
    
    LONG_DEFAULT_CKPT_NAME=$(basename "$LONG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    LONG_DEFAULT_CKPT_NAME=$(sanitize_path "$LONG_DEFAULT_CKPT_NAME")
    
    echo "Found Longformer Default checkpoint from previous run: $LONG_DEFAULT_DIR_REL (Name: $LONG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/17_train_long_default.log"
fi

# 18. Download Longformer Default Checkpoint (Background) + Log
if should_run_step 18; then
    echo "[Step 18/$TOTAL_STEPS] Downloading Longformer Default Checkpoint and Log (background)..." | tee "$LOG_DIR/18_scp_long_default_ckpt.log"
    PID_SCP_LONG_DEFAULT_CKPT=$(start_scp "$LOG_DIR/18_scp_long_default_ckpt.log" "$LONG_DEFAULT_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Default Checkpoint")
    PID_SCP_LONG_DEFAULT_TRAIN_LOG=$(start_log_scp "$LOG_DIR/17_train_long_default.log" "$LOG_DIR/18_scp_long_default_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Default Train Log")
else
    skip_step 18 "Downloading Longformer Default Checkpoint and Log" "$LOG_DIR/18_scp_long_default_ckpt.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_DEFAULT_CKPT=""
    PID_SCP_LONG_DEFAULT_TRAIN_LOG=""
fi

# 19. Run Inference (Longformer Default)
if should_run_step 19; then
    echo "[Step 19/$TOTAL_STEPS] Running Inference (Longformer Default)..." | tee "$LOG_DIR/19_infer_long_default.log"
    log_exec "$LOG_DIR/19_infer_long_default.log" python inference.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --batch $INFER_BATCH_SIZE --mask 0.75
    PID_SCP_LONG_DEFAULT_INFER_LOG=$(start_log_scp "$LOG_DIR/19_infer_long_default.log" "$LOG_DIR/20_intgrad_long_default.log" "$LOCAL_LOG_DEST" "Longformer Default Inference Log")
    echo "Inference (Longformer Default) finished."

    # Delete .npy files after inference
    delete_npy_files "$LONG_DEFAULT_DIR_REL" "$LOG_DIR/19_infer_long_default.log"
else
    skip_step 19 "Running Inference (Longformer Default)" "$LOG_DIR/19_infer_long_default.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_DEFAULT_INFER_LOG=""
fi

# 20. Run Integrated Gradients (Longformer Default) + Start Downloads
if should_run_step 20; then
    echo "[Step 20/$TOTAL_STEPS] Running Integrated Gradients (Longformer Default)..." | tee "$LOG_DIR/20_intgrad_long_default.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/20_intgrad_long_default.log"
    log_exec "$LOG_DIR/20_intgrad_long_default.log" python visualize/int_grad.py --checkpoint "$LONG_DEFAULT_CKPT_NAME" --model long --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/20_intgrad_long_default.log"
    echo "Integrated Gradients (Longformer Default) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$LONG_DEFAULT_DIR_REL" "$LOG_DIR/20_intgrad_long_default.log"

    # Start Downloads for IntGrad and previous Inference log
    echo "[Step 20/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (Longformer Default) (background)..." | tee -a "$LOG_DIR/20_intgrad_long_default.log"
    PID_SCP_LONG_DEFAULT_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/20_intgrad_long_default.log" "$LONG_DEFAULT_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "Longformer Default IntGrad Results")
    PID_SCP_LONG_DEFAULT_INFER_LOG=$(start_log_scp "$LOG_DIR/19_infer_long_default.log" "$LOG_DIR/20_intgrad_long_default.log" "$LOCAL_LOG_DEST" "Longformer Default Inference Log")
else
    skip_step 20 "Running Integrated Gradients (Longformer Default)" "$LOG_DIR/20_intgrad_long_default.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_DEFAULT_INTGRAD_FILES=""
    PID_SCP_LONG_DEFAULT_INFER_LOG=""
fi

# 21. Train Longformer (Pretrained Embeddings)
if should_run_step 21; then
    echo "[Step 21/$TOTAL_STEPS] Training Longformer (Pretrained Embeddings)..." | tee "$LOG_DIR/21_train_long_embedding.log"
    log_exec "$LOG_DIR/21_train_long_embedding.log" python train.py --model long --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/long_embedding_weights.pt
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/21_train_long_embedding.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    LONG_EMBEDDING_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/21_train_long_embedding.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    LONG_EMBEDDING_DIR_REL=$(sanitize_path "$LONG_EMBEDDING_DIR_REL")

    if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not determine Longformer embedding checkpoint directory from training output." | tee -a "$LOG_DIR/21_train_long_embedding.log"
        # Fallback is still useful if script output format changes unexpectedly
        LONG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
        if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/21_train_long_embedding.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/21_train_long_embedding.log"
        fi
    fi
    
    # Verify the directory exists
    if [ ! -d "$LONG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/21_train_long_embedding.log"
        # Try a more aggressive fallback
        LONG_EMBEDDING_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*long*pretrained_emb*" | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ] || [ ! -d "$LONG_EMBEDDING_DIR_REL" ]; then
            echo "ERROR: Could not find any Longformer pretrained embedding checkpoint directory." | tee -a "$LOG_DIR/21_train_long_embedding.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/21_train_long_embedding.log"
        fi
    fi
    
    LONG_EMBEDDING_CKPT_NAME=$(basename "$LONG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    LONG_EMBEDDING_CKPT_NAME=$(sanitize_path "$LONG_EMBEDDING_CKPT_NAME")
    
    echo "Longformer Pretrained Embedding Training finished. Checkpoint directory: $LONG_EMBEDDING_DIR_REL (Name: $LONG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/21_train_long_embedding.log"

    # Delete .npy files after training
    delete_npy_files "$LONG_EMBEDDING_DIR_REL" "$LOG_DIR/21_train_long_embedding.log"
else
    skip_step 21 "Training Longformer (Pretrained Embeddings)" "$LOG_DIR/21_train_long_embedding.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    LONG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
    if [ "$LONG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not find Longformer embedding checkpoint directory from previous runs." | tee -a "$LOG_DIR/21_train_long_embedding.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/21_train_long_embedding.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    LONG_EMBEDDING_DIR_REL=$(sanitize_path "$LONG_EMBEDDING_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$LONG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $LONG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/21_train_long_embedding.log"
        exit 1
    fi
    
    LONG_EMBEDDING_CKPT_NAME=$(basename "$LONG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    LONG_EMBEDDING_CKPT_NAME=$(sanitize_path "$LONG_EMBEDDING_CKPT_NAME")
    
    echo "Found Longformer Embedding checkpoint from previous run: $LONG_EMBEDDING_DIR_REL (Name: $LONG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/21_train_long_embedding.log"
fi

# 22. Download Longformer Embedding Checkpoint (Background) + Logs
if should_run_step 22; then
    echo "[Step 22/$TOTAL_STEPS] Downloading Longformer Embedding Checkpoint and Logs (background)..." | tee "$LOG_DIR/22_scp_long_embedding_ckpt.log"
    PID_SCP_LONG_EMBEDDING_CKPT=$(start_scp "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LONG_EMBEDDING_DIR_REL" "$LOCAL_CKPT_DEST_BASE" "Longformer Embedding Checkpoint")
    PID_SCP_LONG_EMBEDDING_TRAIN_LOG=$(start_log_scp "$LOG_DIR/21_train_long_embedding.log" "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Embedding Train Log")
    PID_SCP_LONG_DEFAULT_INTGRAD_LOG=$(start_log_scp "$LOG_DIR/20_intgrad_long_default.log" "$LOG_DIR/22_scp_long_embedding_ckpt.log" "$LOCAL_LOG_DEST" "Longformer Default IntGrad Log")
else
    skip_step 22 "Downloading Longformer Embedding Checkpoint and Logs" "$LOG_DIR/22_scp_long_embedding_ckpt.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_EMBEDDING_CKPT=""
    PID_SCP_LONG_EMBEDDING_TRAIN_LOG=""
    PID_SCP_LONG_DEFAULT_INTGRAD_LOG=""
fi

# 23. Run Inference (Longformer Embedding)
if should_run_step 23; then
    echo "[Step 23/$TOTAL_STEPS] Running Inference (Longformer Embedding)..." | tee "$LOG_DIR/23_infer_long_embedding.log"
    log_exec "$LOG_DIR/23_infer_long_embedding.log" python inference.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --batch $INFER_BATCH_SIZE --mask 0.75
    PID_SCP_LONG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/23_infer_long_embedding.log" "$LOG_DIR/24_intgrad_long_embedding.log" "$LOCAL_LOG_DEST" "Longformer Embedding Inference Log")
    echo "Inference (Longformer Embedding) finished."

    # Delete .npy files after inference
    delete_npy_files "$LONG_EMBEDDING_DIR_REL" "$LOG_DIR/23_infer_long_embedding.log"
else
    skip_step 23 "Running Inference (Longformer Embedding)" "$LOG_DIR/23_infer_long_embedding.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_EMBEDDING_INFER_LOG=""
fi

# 24. Run Integrated Gradients (Longformer Embedding) + Start Downloads
if should_run_step 24; then
    echo "[Step 24/$TOTAL_STEPS] Running Integrated Gradients (Longformer Embedding)..." | tee "$LOG_DIR/24_intgrad_long_embedding.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/24_intgrad_long_embedding.log"
    log_exec "$LOG_DIR/24_intgrad_long_embedding.log" python visualize/int_grad.py --checkpoint "$LONG_EMBEDDING_CKPT_NAME" --model long --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/24_intgrad_long_embedding.log"
    echo "Integrated Gradients (Longformer Embedding) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$LONG_EMBEDDING_DIR_REL" "$LOG_DIR/24_intgrad_long_embedding.log"

    # Start Downloads for IntGrad and previous Inference log
    echo "[Step 24/$TOTAL_STEPS] Downloading IntGrad results & Inference Log (Longformer Embedding) (background)..." | tee -a "$LOG_DIR/24_intgrad_long_embedding.log"
    PID_SCP_LONG_EMBEDDING_INTGRAD_FILES=$(start_int_grad_scp "$LOG_DIR/24_intgrad_long_embedding.log" "$LONG_EMBEDDING_DIR_REL" "$LOCAL_INT_GRAD_DEST_BASE" "Longformer Embedding IntGrad Results")
    PID_SCP_LONG_EMBEDDING_INFER_LOG=$(start_log_scp "$LOG_DIR/23_infer_long_embedding.log" "$LOG_DIR/24_intgrad_long_embedding.log" "$LOCAL_LOG_DEST" "Longformer Embedding Inference Log")
else
    skip_step 24 "Running Integrated Gradients (Longformer Embedding)" "$LOG_DIR/24_intgrad_long_embedding.log"
    # If skipping, initialize variables to prevent errors
    PID_SCP_LONG_EMBEDDING_INTGRAD_FILES=""
    PID_SCP_LONG_EMBEDDING_INFER_LOG=""
fi

# --- Wait for all background downloads ---
echo "--- Waiting for background downloads to complete ---" | tee "$LOG_DIR/99_wait_downloads.log"
FAIL_COUNT=0
PIDS_TO_WAIT=(
    $PID_SCP_BIG_EMB_FILE
    $PID_SCP_BIG_EMB_LOG
    $PID_SCP_BIG_DEFAULT_CKPT
    $PID_SCP_BIG_DEFAULT_LOG
    $PID_SCP_BIG_DEFAULT_INFER_LOG
    $PID_SCP_BIG_DEFAULT_INTGRAD_FILES
    $PID_SCP_BIG_DEFAULT_INTGRAD_LOG
    $PID_SCP_BIG_EMBEDDING_CKPT
    $PID_SCP_BIG_EMBEDDING_TRAIN_LOG
    $PID_SCP_BIG_EMBEDDING_INFER_LOG
    $PID_SCP_BIG_EMBEDDING_INTGRAD_FILES
    $PID_SCP_BIG_EMBEDDING_INTGRAD_LOG

    $PID_SCP_LONG_EMB_FILE
    $PID_SCP_LONG_PRETRAIN_LOG
    $PID_SCP_LONG_DEFAULT_CKPT
    $PID_SCP_LONG_DEFAULT_TRAIN_LOG
    $PID_SCP_LONG_DEFAULT_INFER_LOG
    $PID_SCP_LONG_DEFAULT_INTGRAD_FILES
    $PID_SCP_LONG_DEFAULT_INTGRAD_LOG

    $PID_SCP_LONG_EMBEDDING_CKPT
    $PID_SCP_LONG_EMBEDDING_TRAIN_LOG
    $PID_SCP_LONG_EMBEDDING_INFER_LOG
    $PID_SCP_LONG_EMBEDDING_INTGRAD_FILES
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