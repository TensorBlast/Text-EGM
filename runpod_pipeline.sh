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

# Default batch sizes
TRAIN_BATCH_SIZE=8
INFER_BATCH_SIZE=16
RESUME_FROM=1  # Default to starting from the first step

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
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
      # Unknown option
      echo "Unknown option: $1"
      echo "Usage: $0 [--train-batch SIZE] [--infer-batch SIZE] [--resume-from STEP]"
      exit 1
      ;;
  esac
done

echo "Using training batch size: $TRAIN_BATCH_SIZE"
echo "Using inference batch size: $INFER_BATCH_SIZE"
echo "Resuming from step: $RESUME_FROM"

# Total number of steps in the pipeline - define this early
TOTAL_STEPS=18 # Update this if steps are added or removed

# Detect if running in Docker container
IN_DOCKER=false
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER=true
    echo "Docker environment detected."
fi

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
    echo "3: Preprocessing"
    echo "4: Pretrain BigBird Embeddings"
    echo "5: Archive and Send BigBird Embeddings"
    echo "6: Train BigBird (Default)"
    echo "7: Run Inference (BigBird Default)"
    echo "8: Run Integrated Gradients (BigBird Default)"
    echo "9: Train BigBird (Pretrained Embeddings)"
    echo "10: Run Inference (BigBird Embedding)"
    echo "11: Run Integrated Gradients (BigBird Embedding)"
    echo "12: Pretrain Longformer Embeddings"
    echo "13: Train Longformer (Default)"
    echo "14: Run Inference (Longformer Default)"
    echo "15: Run Integrated Gradients (Longformer Default)"
    echo "16: Train Longformer (Pretrained Embeddings)"
    echo "17: Run Inference (Longformer Embedding)"
    echo "18: Run Integrated Gradients (Longformer Embedding)"
    exit 0
fi

# Validate RESUME_FROM is a number
if ! [[ "$RESUME_FROM" =~ ^[0-9]+$ ]]; then
    # Skip validation if RESUME_FROM is "list"
    if [ "$RESUME_FROM" != "list" ]; then
        echo "ERROR: --resume-from must be a number (1-16) or 'list'"
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

# Executes a command, logging stdout and stderr to a file and also displaying output
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
    return 0
}

# Creates a zip archive of specified files or directories
create_archive() {
    local archive_name="$1"
    local source_path="$2"
    local description="$3"
    local log_file="$4"
    local archive_path="${ARCHIVE_DIR}/${archive_name}.zip"
    
    # Normalize paths to avoid double slashes
    source_path=$(echo "$source_path" | sed 's|//|/|g')
    archive_path=$(echo "$archive_path" | sed 's|//|/|g')
    
    echo "Creating archive: ${archive_name}.zip from ${source_path}" | tee -a "$log_file"
    
    # Create the archive directory if it doesn't exist
    mkdir -p "${ARCHIVE_DIR}" || {
        echo "WARNING: Failed to create archive directory ${ARCHIVE_DIR}. Continuing..." | tee -a "$log_file"
        return 0
    }
    
    # If source is a directory with specific file patterns to include
    if [[ "$source_path" == *"PATTERN:"* ]]; then
        local dir_path=$(echo "$source_path" | cut -d':' -f2)
        local patterns=$(echo "$source_path" | cut -d':' -f3-)
        
        # Normalize dir_path
        dir_path=$(echo "$dir_path" | sed 's|//|/|g')
        
        # Create a temporary directory for the files to zip
        local temp_dir="${ARCHIVE_DIR}/temp_${archive_name}"
        mkdir -p "$temp_dir" || {
            echo "WARNING: Failed to create temp directory ${temp_dir}. Continuing..." | tee -a "$log_file"
            return 0
        }
        
        # Copy only the specified file patterns to the temp directory
        local copy_success=false
        for pattern in $patterns; do
            if find "$dir_path" -maxdepth 1 -name "$pattern" -exec cp {} "$temp_dir/" \; 2>/dev/null; then
                copy_success=true
            else
                echo "WARNING: No files matching pattern $pattern found in $dir_path" | tee -a "$log_file"
            fi
        done
        
        if [ "$copy_success" = false ]; then
            echo "WARNING: No files copied for any patterns. Archive may be empty." | tee -a "$log_file"
        fi
        
        # Zip the temp directory contents
        if ! (cd "$temp_dir" && zip -r "$archive_path" . >> "$log_file" 2>&1); then
            echo "WARNING: Failed to create zip archive ${archive_path}. Continuing..." | tee -a "$log_file"
        fi
        
        # Clean up the temp directory
        rm -rf "$temp_dir" 2>/dev/null || echo "WARNING: Failed to remove temp directory ${temp_dir}" | tee -a "$log_file"
    else
        # Check if source exists before zipping
        if [ ! -e "$source_path" ]; then
            echo "WARNING: Source path $source_path does not exist. Cannot create archive." | tee -a "$log_file"
            return 0
        fi
        
        # Special handling for single files vs directories
        if [ -f "$source_path" ]; then
            # For a single file, use a simpler approach:
            # 1. Get the directory and filename
            local source_dir=$(dirname "$source_path")
            local source_file=$(basename "$source_path")
            
            # 2. Change to the directory and zip just the file (to avoid full path in archive)
            if ! (cd "$source_dir" && zip "$archive_path" "$source_file" >> "$log_file" 2>&1); then
                echo "WARNING: Failed to create archive for single file ${source_path}. Continuing..." | tee -a "$log_file"
                return 0
            fi
        else
            # Standard zip operation for directories
            if ! zip -r "$archive_path" "$source_path" >> "$log_file" 2>&1; then
                echo "WARNING: Failed to create archive ${archive_path}. Continuing..." | tee -a "$log_file"
                return 0
            fi
        fi
    fi
    
    # Check if archive was created successfully
    if [ -f "$archive_path" ]; then
        echo "Archive created successfully: ${archive_path}" | tee -a "$log_file"
    else
        echo "WARNING: Archive file ${archive_path} does not exist after creation attempt." | tee -a "$log_file"
    fi
    
    return 0  # Always return success to prevent script termination
}

# Sends a file using runpodctl and logs the receive command
send_with_runpodctl() {
    local archive_path="$1"
    local description="$2"
    local log_file="$3"
    
    echo "Sending archive: ${archive_path} (${description})" | tee -a "$log_file"
    
    # Check if the archive exists before attempting to send
    if [ ! -f "$archive_path" ]; then
        echo "WARNING: Archive file ${archive_path} does not exist. Cannot send." | tee -a "$log_file"
        return 0
    fi
    
    # Check if runpodctl exists
    if ! command -v runpodctl &> /dev/null; then
        echo "WARNING: runpodctl command not found. Cannot send archive." | tee -a "$log_file"
        return 0
    fi
    
    # Create a dedicated log file for this specific transfer
    local transfer_log="${LOG_DIR}/transfer_${description// /_}_$(date +%s).log"
    
    # Run runpodctl in the background with its output going to a dedicated log file
    runpodctl send "$archive_path" > "$transfer_log" 2>&1 &
    local send_pid=$!
    
    # Wait a short while to capture the initial output with the code
    sleep 3
    
    # Read the current content of the log to extract the receive command
    if [ -f "$transfer_log" ]; then
        local log_content=$(cat "$transfer_log" 2>/dev/null || echo "Error reading log file")
        echo "Transfer started in background (PID: $send_pid). Initial output:" | tee -a "$log_file"
        echo "$log_content" | tee -a "$log_file"
        
        # Extract the receive command and code
        local receive_code=$(echo "$log_content" | grep -oE "[0-9]+-[a-z]+-[a-z]+-[a-z]+-?[0-9]*" | head -1 || echo "")
        
        if [ -n "$receive_code" ]; then
            echo -e "\n\n============ IMPORTANT: RUNPODCTL RECEIVE COMMAND ============" | tee -a "$log_file"
            echo "To download ${description}, run this command on your local machine:" | tee -a "$log_file"
            echo "runpodctl receive $receive_code" | tee -a "$log_file"
            echo "============================================================\n" | tee -a "$log_file"
            
            # Save the receive command to a dedicated file for easy reference
            echo "runpodctl receive $receive_code" > "${LOG_DIR}/receive_${description// /_}.txt" 2>/dev/null || echo "WARNING: Failed to save receive command to file" | tee -a "$log_file"
            echo "The receive command has also been saved to: ${LOG_DIR}/receive_${description// /_}.txt" | tee -a "$log_file"
        else
            echo "WARNING: Could not extract receive code from initial output." | tee -a "$log_file"
            echo "Please check the transfer log for the complete output: $transfer_log" | tee -a "$log_file"
        fi
    else
        echo "WARNING: Transfer log file $transfer_log could not be created or accessed." | tee -a "$log_file"
    fi
    
    echo "File transfer initiated in background. Continuing with pipeline..." | tee -a "$log_file"
    return 0  # Always return success to prevent script termination
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
echo "Starting Full Text-EGM Pipeline on RunPod Instance"
echo "=================================================="

# Create log and archive directories with absolute paths
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" # Ensure directories exist early
chmod 755 "$LOG_DIR" "$ARCHIVE_DIR" # Ensure proper permissions

# Log the directories being used
echo "Using log directory: $LOG_DIR"
echo "Using archive directory: $ARCHIVE_DIR"
echo "Using script directory: $SCRIPT_DIR"

date > "$LOG_DIR/00_start_time.log"

# 1. System Prep
if should_run_step 1; then
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

# 3. Preprocessing
if should_run_step 3; then
    echo "[Step 3/$TOTAL_STEPS] Running Preprocessing..." | tee "$LOG_DIR/03_preprocess.log"
    # Check if preprocessed data files already exist
    DATA_DIR="../data"
    if [ -f "${DATA_DIR}/train_intra.npy" ] && [ -f "${DATA_DIR}/val_intra.npy" ] && [ -f "${DATA_DIR}/test_intra.npy" ]; then
        echo "Preprocessed data files already exist in ${DATA_DIR}. Skipping preprocessing." | tee -a "$LOG_DIR/03_preprocess.log"
    else
        echo "Preprocessed data files not found. Running preprocessing..." | tee -a "$LOG_DIR/03_preprocess.log"
        log_exec "$LOG_DIR/03_preprocess.log" python preprocess/preprocess_intra.py
        echo "Preprocessing finished."
    fi
else
    skip_step 3 "Preprocessing" "$LOG_DIR/03_preprocess.log"
fi

# --- BigBird Model Pipeline ---
echo "--- Starting BigBird Pipeline ---"

# 4. Pretrain Embeddings (BigBird)
if should_run_step 4; then
    echo "[Step 4/$TOTAL_STEPS] Pretraining BigBird Embeddings..." | tee "$LOG_DIR/04_pretrain_big.log"
    BIG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/big_embedding_weights.pt"
    if [ -f "$BIG_EMBEDDING_FILE" ]; then
        echo "BigBird embedding file already exists at $BIG_EMBEDDING_FILE. Skipping pretraining." | tee -a "$LOG_DIR/04_pretrain_big.log"
    else
        log_exec "$LOG_DIR/04_pretrain_big.log" python pretrain_embeddings.py --model big --epochs 30 --batch $INFER_BATCH_SIZE --output_dir ./"$PRETRAINED_EMB_DIR"
        echo "BigBird Embedding Pretraining finished."
    fi
else
    skip_step 4 "Pretraining BigBird Embeddings" "$LOG_DIR/04_pretrain_big.log"
fi

# 5. Archive and Send BigBird Embeddings
if should_run_step 5; then
    echo "[Step 5/$TOTAL_STEPS] Archiving and Sending BigBird Embeddings..." | tee "$LOG_DIR/05_archive_big_emb.log"
    create_archive "Big_Embeddings" "./$PRETRAINED_EMB_DIR/big_embedding_weights.pt" "BigBird Embeddings" "$LOG_DIR/05_archive_big_emb.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Embeddings.zip" "BigBird Embeddings" "$LOG_DIR/05_archive_big_emb.log"
else
    skip_step 5 "Archiving and Sending BigBird Embeddings" "$LOG_DIR/05_archive_big_emb.log"
fi

# 6. Train BigBird (Default)
if should_run_step 6; then
    echo "[Step 6/$TOTAL_STEPS] Training BigBird (Default)..." | tee "$LOG_DIR/06_train_big_default.log"
    # Run training directly using log_exec
    log_exec "$LOG_DIR/06_train_big_default.log" python train.py --model big --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script (BigBird Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/06_train_big_default.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    BIG_DEFAULT_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/06_train_big_default.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")

    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not determine BigBird default checkpoint directory from training output." | tee -a "$LOG_DIR/06_train_big_default.log"
        # Fallback is still useful if script output format changes unexpectedly
        BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
        if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/06_train_big_default.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/06_train_big_default.log"
             # Sanitize the fallback path too
             BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")
        fi
    fi

    # Verify the directory exists
    if [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/06_train_big_default.log"
        # Try a more aggressive fallback
        BIG_DEFAULT_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*big*" | grep -v 'pretrained_emb' | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ] || [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
            echo "ERROR: Could not find any BigBird default checkpoint directory." | tee -a "$LOG_DIR/06_train_big_default.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/06_train_big_default.log"
        fi
    fi
    
    BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_DEFAULT_CKPT_NAME=$(sanitize_path "$BIG_DEFAULT_CKPT_NAME")
    
    echo "BigBird Default Training finished. Checkpoint directory: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/06_train_big_default.log"

    # Delete .npy files after training
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/06_train_big_default.log"
else
    skip_step 6 "Training BigBird (Default)" "$LOG_DIR/06_train_big_default.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    BIG_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
    if [ "$BIG_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not find BigBird default checkpoint directory from previous runs." | tee -a "$LOG_DIR/06_train_big_default.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/06_train_big_default.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    BIG_DEFAULT_DIR_REL=$(sanitize_path "$BIG_DEFAULT_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$BIG_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $BIG_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/06_train_big_default.log"
        exit 1
    fi
    
    BIG_DEFAULT_CKPT_NAME=$(basename "$BIG_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_DEFAULT_CKPT_NAME=$(sanitize_path "$BIG_DEFAULT_CKPT_NAME")
    
    echo "Found BigBird Default checkpoint from previous run: $BIG_DEFAULT_DIR_REL (Name: $BIG_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/06_train_big_default.log"
fi

# 7. Run Inference (BigBird Default)
if should_run_step 7; then
    echo "[Step 7/$TOTAL_STEPS] Running Inference (BigBird Default)..." | tee "$LOG_DIR/07_infer_big_default.log"
    log_exec "$LOG_DIR/07_infer_big_default.log" python inference.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --batch $INFER_BATCH_SIZE --mask 0.75
    echo "Inference (BigBird Default) finished."

    # Delete .npy files after inference
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/07_infer_big_default.log"

    # Archive and send inference log
    echo "Archiving and sending BigBird Default inference log..." | tee -a "$LOG_DIR/07_infer_big_default.log"
    echo "Log file to archive: $LOG_DIR/07_infer_big_default.log" | tee -a "$LOG_DIR/07_infer_big_default.log"
    if [ ! -f "$LOG_DIR/07_infer_big_default.log" ]; then
        echo "WARNING: Inference log file does not exist at $LOG_DIR/07_infer_big_default.log" | tee -a "$LOG_DIR/07_infer_big_default.log"
        # Create an empty log file if it doesn't exist to prevent further errors
        touch "$LOG_DIR/07_infer_big_default.log"
    fi
    ls -la "$LOG_DIR/" | grep "07_infer_big_default.log" | tee -a "$LOG_DIR/07_infer_big_default.log"
    create_archive "Big_Default_Inference_Log" "$LOG_DIR/07_infer_big_default.log" "BigBird Default Inference Log" "$LOG_DIR/07_infer_big_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Default_Inference_Log.zip" "BigBird Default Inference Log" "$LOG_DIR/07_infer_big_default.log"
else
    skip_step 7 "Running Inference (BigBird Default)" "$LOG_DIR/07_infer_big_default.log"
fi

# 8. Run Integrated Gradients (BigBird Default) + Archive and Send
if should_run_step 8; then
    echo "[Step 8/$TOTAL_STEPS] Running Integrated Gradients (BigBird Default)..." | tee "$LOG_DIR/08_intgrad_big_default.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/08_intgrad_big_default.log"
    log_exec "$LOG_DIR/08_intgrad_big_default.log" python visualize/int_grad.py --checkpoint "$BIG_DEFAULT_CKPT_NAME" --model big --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/08_intgrad_big_default.log"
    echo "Integrated Gradients (BigBird Default) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$BIG_DEFAULT_DIR_REL" "$LOG_DIR/08_intgrad_big_default.log"

    # Create archive of IntGrad results and checkpoint
    echo "[Step 8/$TOTAL_STEPS] Archiving and Sending BigBird Default Results..." | tee -a "$LOG_DIR/08_intgrad_big_default.log"
    create_archive "Big_Default_IntGrad" "PATTERN:$BIG_DEFAULT_DIR_REL:*.png *.npy" "BigBird Default IntGrad Results" "$LOG_DIR/08_intgrad_big_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Default_IntGrad.zip" "BigBird Default IntGrad Results" "$LOG_DIR/08_intgrad_big_default.log"
    create_archive "Big_Default_Checkpoint" "$BIG_DEFAULT_DIR_REL" "BigBird Default Checkpoint" "$LOG_DIR/08_intgrad_big_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Default_Checkpoint.zip" "BigBird Default Checkpoint" "$LOG_DIR/08_intgrad_big_default.log"
else
    skip_step 8 "Running Integrated Gradients (BigBird Default)" "$LOG_DIR/08_intgrad_big_default.log"
fi

# 9. Train BigBird (Pretrained Embeddings)
if should_run_step 9; then
    echo "[Step 9/$TOTAL_STEPS] Training BigBird (Pretrained Embeddings)..." | tee "$LOG_DIR/09_train_big_embedding.log"
    # Run training directly using log_exec
    log_exec "$LOG_DIR/09_train_big_embedding.log" python train.py --model big --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/big_embedding_weights.pt
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script (BigBird Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/09_train_big_embedding.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    BIG_EMBEDDING_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/09_train_big_embedding.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")

    if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not determine BigBird embedding checkpoint directory from training output." | tee -a "$LOG_DIR/09_train_big_embedding.log"
        # Fallback is still useful
        BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/09_train_big_embedding.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
             # Sanitize the fallback path too
             BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")
        fi
    fi
    
    # Verify the directory exists
    if [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
        # Try a more aggressive fallback
        BIG_EMBEDDING_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*big*" | grep -v 'pretrained_emb' | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ] || [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
            echo "ERROR: Could not find any BigBird pretrained embedding checkpoint directory." | tee -a "$LOG_DIR/09_train_big_embedding.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
        fi
    fi
    
    BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_EMBEDDING_CKPT_NAME=$(sanitize_path "$BIG_EMBEDDING_CKPT_NAME")
    
    echo "BigBird Pretrained Embedding Training finished. Checkpoint directory: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/09_train_big_embedding.log"

    # Delete .npy files after training
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/09_train_big_embedding.log"
else
    skip_step 9 "Training BigBird (Pretrained Embeddings)" "$LOG_DIR/09_train_big_embedding.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    BIG_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_big_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
    if [ "$BIG_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not find BigBird embedding checkpoint directory from previous runs." | tee -a "$LOG_DIR/09_train_big_embedding.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/09_train_big_embedding.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    BIG_EMBEDDING_DIR_REL=$(sanitize_path "$BIG_EMBEDDING_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$BIG_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $BIG_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/09_train_big_embedding.log"
        exit 1
    fi
    
    BIG_EMBEDDING_CKPT_NAME=$(basename "$BIG_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    BIG_EMBEDDING_CKPT_NAME=$(sanitize_path "$BIG_EMBEDDING_CKPT_NAME")
    
    echo "Found BigBird Embedding checkpoint from previous run: $BIG_EMBEDDING_DIR_REL (Name: $BIG_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/09_train_big_embedding.log"
fi

# 10. Run Inference (BigBird Embedding)
if should_run_step 10; then
    echo "[Step 10/$TOTAL_STEPS] Running Inference (BigBird Embedding)..." | tee "$LOG_DIR/10_infer_big_embedding.log"
    log_exec "$LOG_DIR/10_infer_big_embedding.log" python inference.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --batch $INFER_BATCH_SIZE --mask 0.75
    echo "Inference (BigBird Embedding) finished."

    # Delete .npy files after inference
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/10_infer_big_embedding.log"

    # Archive and send inference log
    echo "Archiving and sending BigBird Embedding inference log..." | tee -a "$LOG_DIR/10_infer_big_embedding.log"
    create_archive "Big_Embedding_Inference_Log" "$LOG_DIR/10_infer_big_embedding.log" "BigBird Embedding Inference Log" "$LOG_DIR/10_infer_big_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Embedding_Inference_Log.zip" "BigBird Embedding Inference Log" "$LOG_DIR/10_infer_big_embedding.log"
else
    skip_step 10 "Running Inference (BigBird Embedding)" "$LOG_DIR/10_infer_big_embedding.log"
fi

# 11. Run Integrated Gradients (BigBird Embedding) + Archive and Send
if should_run_step 11; then
    echo "[Step 11/$TOTAL_STEPS] Running Integrated Gradients (BigBird Embedding)..." | tee "$LOG_DIR/11_intgrad_big_embedding.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/11_intgrad_big_embedding.log"
    log_exec "$LOG_DIR/11_intgrad_big_embedding.log" python visualize/int_grad.py --checkpoint "$BIG_EMBEDDING_CKPT_NAME" --model big --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/11_intgrad_big_embedding.log"
    echo "Integrated Gradients (BigBird Embedding) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$BIG_EMBEDDING_DIR_REL" "$LOG_DIR/11_intgrad_big_embedding.log"

    # Create archive of IntGrad results and checkpoint
    echo "[Step 11/$TOTAL_STEPS] Archiving and Sending BigBird Embedding Results..." | tee -a "$LOG_DIR/11_intgrad_big_embedding.log"
    create_archive "Big_Emb_IntGrad" "PATTERN:$BIG_EMBEDDING_DIR_REL:*.png *.npy" "BigBird Embedding IntGrad Results" "$LOG_DIR/11_intgrad_big_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Emb_IntGrad.zip" "BigBird Embedding IntGrad Results" "$LOG_DIR/11_intgrad_big_embedding.log"
    create_archive "Big_Emb_Checkpoint" "$BIG_EMBEDDING_DIR_REL" "BigBird Embedding Checkpoint" "$LOG_DIR/11_intgrad_big_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Big_Emb_Checkpoint.zip" "BigBird Embedding Checkpoint" "$LOG_DIR/11_intgrad_big_embedding.log"
else
    skip_step 11 "Running Integrated Gradients (BigBird Embedding)" "$LOG_DIR/11_intgrad_big_embedding.log"
fi

# --- Longformer Model Pipeline ---
echo "--- Starting Longformer Pipeline ---"

# 12. Pretrain Embeddings (Longformer)
if should_run_step 12; then
    echo "[Step 12/$TOTAL_STEPS] Pretraining Longformer Embeddings..." | tee "$LOG_DIR/12_pretrain_long.log"
    LONG_EMBEDDING_FILE="./$PRETRAINED_EMB_DIR/long_embedding_weights.pt"
    if [ -f "$LONG_EMBEDDING_FILE" ]; then
        echo "Longformer embedding file already exists at $LONG_EMBEDDING_FILE. Skipping pretraining." | tee -a "$LOG_DIR/12_pretrain_long.log"
    else
        log_exec "$LOG_DIR/12_pretrain_long.log" python pretrain_embeddings.py --model long --epochs 30 --batch $INFER_BATCH_SIZE --output_dir ./"$PRETRAINED_EMB_DIR"
        echo "Longformer Embedding Pretraining finished."
    fi

    # Archive and Send Longformer Embeddings
    echo "[Step 12/$TOTAL_STEPS] Archiving and Sending Longformer Embeddings..." | tee -a "$LOG_DIR/12_pretrain_long.log"
    create_archive "Long_Embeddings" "./$PRETRAINED_EMB_DIR/long_embedding_weights.pt" "Longformer Embeddings" "$LOG_DIR/12_pretrain_long.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Embeddings.zip" "Longformer Embeddings" "$LOG_DIR/12_pretrain_long.log"
else
    skip_step 12 "Pretraining Longformer Embeddings" "$LOG_DIR/12_pretrain_long.log"
fi

# 13. Train Longformer (Default)
if should_run_step 13; then
    echo "[Step 13/$TOTAL_STEPS] Training Longformer (Default)..." | tee "$LOG_DIR/13_train_long_default.log"
    # Run training directly using log_exec
    log_exec "$LOG_DIR/13_train_long_default.log" python train.py --model long --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script (Longformer Default) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/13_train_long_default.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    LONGFORMER_DEFAULT_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/13_train_long_default.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    LONGFORMER_DEFAULT_DIR_REL=$(sanitize_path "$LONGFORMER_DEFAULT_DIR_REL")

    if [ "$LONGFORMER_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not determine Longformer default checkpoint directory from training output." | tee -a "$LOG_DIR/13_train_long_default.log"
        # Fallback is still useful if script output format changes unexpectedly
        LONGFORMER_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND") # Exclude pretrained_emb
        if [ "$LONGFORMER_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_DEFAULT_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/13_train_long_default.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $LONGFORMER_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
             # Sanitize the fallback path too
             LONGFORMER_DEFAULT_DIR_REL=$(sanitize_path "$LONGFORMER_DEFAULT_DIR_REL")
        fi
    fi

    # Verify the directory exists
    if [ ! -d "$LONGFORMER_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $LONGFORMER_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
        # Try a more aggressive fallback
        LONGFORMER_DEFAULT_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*long*" | grep -v 'pretrained_emb' | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$LONGFORMER_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_DEFAULT_DIR_REL" ] || [ ! -d "$LONGFORMER_DEFAULT_DIR_REL" ]; then
            echo "ERROR: Could not find any Longformer default checkpoint directory." | tee -a "$LOG_DIR/13_train_long_default.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $LONGFORMER_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
        fi
    fi
    
    LONGFORMER_DEFAULT_CKPT_NAME=$(basename "$LONGFORMER_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    LONGFORMER_DEFAULT_CKPT_NAME=$(sanitize_path "$LONGFORMER_DEFAULT_CKPT_NAME")
    
    echo "Longformer Default Training finished. Checkpoint directory: $LONGFORMER_DEFAULT_DIR_REL (Name: $LONGFORMER_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/13_train_long_default.log"

    # Delete .npy files after training
    delete_npy_files "$LONGFORMER_DEFAULT_DIR_REL" "$LOG_DIR/13_train_long_default.log"
else
    skip_step 13 "Training Longformer (Default)" "$LOG_DIR/13_train_long_default.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    LONGFORMER_DEFAULT_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_* | grep -v 'pretrained_emb' | head -n 1 || echo "NOT_FOUND")
    if [ "$LONGFORMER_DEFAULT_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Could not find Longformer default checkpoint directory from previous runs." | tee -a "$LOG_DIR/13_train_long_default.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/13_train_long_default.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    LONGFORMER_DEFAULT_DIR_REL=$(sanitize_path "$LONGFORMER_DEFAULT_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$LONGFORMER_DEFAULT_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $LONGFORMER_DEFAULT_DIR_REL" | tee -a "$LOG_DIR/13_train_long_default.log"
        exit 1
    fi
    
    LONGFORMER_DEFAULT_CKPT_NAME=$(basename "$LONGFORMER_DEFAULT_DIR_REL")
    # Sanitize the checkpoint name too
    LONGFORMER_DEFAULT_CKPT_NAME=$(sanitize_path "$LONGFORMER_DEFAULT_CKPT_NAME")
    
    echo "Found Longformer Default checkpoint from previous run: $LONGFORMER_DEFAULT_DIR_REL (Name: $LONGFORMER_DEFAULT_CKPT_NAME)" | tee -a "$LOG_DIR/13_train_long_default.log"
fi

# 14. Run Inference (Longformer Default)
if should_run_step 14; then
    echo "[Step 14/$TOTAL_STEPS] Running Inference (Longformer Default)..." | tee "$LOG_DIR/14_infer_long_default.log"
    log_exec "$LOG_DIR/14_infer_long_default.log" python inference.py --checkpoint "$LONGFORMER_DEFAULT_CKPT_NAME" --model long --batch $INFER_BATCH_SIZE --mask 0.75
    echo "Inference (Longformer Default) finished."

    # Delete .npy files after inference
    delete_npy_files "$LONGFORMER_DEFAULT_DIR_REL" "$LOG_DIR/14_infer_long_default.log"

    # Archive and send inference log
    echo "Archiving and sending Longformer Default inference log..." | tee -a "$LOG_DIR/14_infer_long_default.log"
    create_archive "Long_Default_Inference_Log" "$LOG_DIR/14_infer_long_default.log" "Longformer Default Inference Log" "$LOG_DIR/14_infer_long_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Default_Inference_Log.zip" "Longformer Default Inference Log" "$LOG_DIR/14_infer_long_default.log"
else
    skip_step 14 "Running Inference (Longformer Default)" "$LOG_DIR/14_infer_long_default.log"
fi

# 15. Run Integrated Gradients (Longformer Default) + Archive and Send
if should_run_step 15; then
    echo "[Step 15/$TOTAL_STEPS] Running Integrated Gradients (Longformer Default)..." | tee "$LOG_DIR/15_intgrad_long_default.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/15_intgrad_long_default.log"
    log_exec "$LOG_DIR/15_intgrad_long_default.log" python visualize/int_grad.py --checkpoint "$LONGFORMER_DEFAULT_CKPT_NAME" --model long --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/15_intgrad_long_default.log"
    echo "Integrated Gradients (Longformer Default) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$LONGFORMER_DEFAULT_DIR_REL" "$LOG_DIR/15_intgrad_long_default.log"

    # Create archive of IntGrad results and checkpoint
    echo "[Step 15/$TOTAL_STEPS] Archiving and Sending Longformer Default Results..." | tee -a "$LOG_DIR/15_intgrad_long_default.log"
    create_archive "Long_Default_IntGrad" "PATTERN:$LONGFORMER_DEFAULT_DIR_REL:*.png *.npy" "Longformer Default IntGrad Results" "$LOG_DIR/15_intgrad_long_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Default_IntGrad.zip" "Longformer Default IntGrad Results" "$LOG_DIR/15_intgrad_long_default.log"
    create_archive "Long_Default_Checkpoint" "$LONGFORMER_DEFAULT_DIR_REL" "Longformer Default Checkpoint" "$LOG_DIR/15_intgrad_long_default.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Default_Checkpoint.zip" "Longformer Default Checkpoint" "$LOG_DIR/15_intgrad_long_default.log"
else
    skip_step 15 "Running Integrated Gradients (Longformer Default)" "$LOG_DIR/15_intgrad_long_default.log"
fi

# 16. Train Longformer (Pretrained Embeddings)
if should_run_step 16; then
    echo "[Step 16/$TOTAL_STEPS] Training Longformer (Pretrained Embeddings)..." | tee "$LOG_DIR/16_train_long_embedding.log"
    # Run training directly using log_exec
    log_exec "$LOG_DIR/16_train_long_embedding.log" python train.py --model long --epochs 20 --batch $TRAIN_BATCH_SIZE --no-use_ce --pretrained_embeddings ./"$PRETRAINED_EMB_DIR"/long_embedding_weights.pt
    PYTHON_EXIT_CODE=$?
    if [ $PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python training script (Longformer Embedding) failed with exit code $PYTHON_EXIT_CODE." | tee -a "$LOG_DIR/16_train_long_embedding.log"
        exit $PYTHON_EXIT_CODE
    fi

    # Extract directory name from the log file
    LONGFORMER_EMBEDDING_DIR_REL=$(grep -Eo 'Directory (created|already exists): \./[^ ]+' "$LOG_DIR/16_train_long_embedding.log" | sed -E 's/Directory (created|already exists): \.\///' | tail -1 || echo "NOT_FOUND")

    # Sanitize the path to remove any problematic characters
    LONGFORMER_EMBEDDING_DIR_REL=$(sanitize_path "$LONGFORMER_EMBEDDING_DIR_REL")

    if [ "$LONGFORMER_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not determine Longformer pretrained embedding checkpoint directory from training output." | tee -a "$LOG_DIR/16_train_long_embedding.log"
        # Fallback is still useful if script output format changes unexpectedly
        LONGFORMER_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
        if [ "$LONGFORMER_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
             echo "ERROR: Fallback directory search also failed." | tee -a "$LOG_DIR/16_train_long_embedding.log"
             exit 1
        else
             echo "WARNING: Used fallback to find checkpoint directory: $LONGFORMER_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/16_train_long_embedding.log"
             # Sanitize the fallback path too
             LONGFORMER_EMBEDDING_DIR_REL=$(sanitize_path "$LONGFORMER_EMBEDDING_DIR_REL")
        fi
    fi

    # Verify the directory exists
    if [ ! -d "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory does not exist: $LONGFORMER_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/16_train_long_embedding.log"
        # Try a more aggressive fallback
        LONGFORMER_EMBEDDING_DIR_REL=$(find runs/checkpoint -maxdepth 1 -type d -name "*long*pretrained_emb*" | sort -r | head -n 1 || echo "NOT_FOUND")
        if [ "$LONGFORMER_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_EMBEDDING_DIR_REL" ] || [ ! -d "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
            echo "ERROR: Could not find any Longformer pretrained embedding checkpoint directory." | tee -a "$LOG_DIR/16_train_long_embedding.log"
            exit 1
        else
            echo "WARNING: Using alternative checkpoint directory: $LONGFORMER_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/16_train_long_embedding.log"
        fi
    fi
    
    LONGFORMER_EMBEDDING_CKPT_NAME=$(basename "$LONGFORMER_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    LONGFORMER_EMBEDDING_CKPT_NAME=$(sanitize_path "$LONGFORMER_EMBEDDING_CKPT_NAME")
    
    echo "Longformer Pretrained Embedding Training finished. Checkpoint directory: $LONGFORMER_EMBEDDING_DIR_REL (Name: $LONGFORMER_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/16_train_long_embedding.log"

    # Delete .npy files after training
    delete_npy_files "$LONGFORMER_EMBEDDING_DIR_REL" "$LOG_DIR/16_train_long_embedding.log"
else
    skip_step 16 "Training Longformer (Pretrained Embeddings)" "$LOG_DIR/16_train_long_embedding.log"
    # If skipping, we need to determine the checkpoint directory from previous runs
    LONGFORMER_EMBEDDING_DIR_REL=$(ls -td runs/checkpoint/saved_best_*_long_*pretrained_emb* | head -n 1 || echo "NOT_FOUND")
    if [ "$LONGFORMER_EMBEDDING_DIR_REL" == "NOT_FOUND" ] || [ -z "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Could not find Longformer embedding checkpoint directory from previous runs." | tee -a "$LOG_DIR/16_train_long_embedding.log"
        echo "Cannot continue without checkpoint. Please run without --resume-from option first." | tee -a "$LOG_DIR/16_train_long_embedding.log"
        exit 1
    fi
    
    # Sanitize the directory path to remove problematic characters
    LONGFORMER_EMBEDDING_DIR_REL=$(sanitize_path "$LONGFORMER_EMBEDDING_DIR_REL")
    
    # Verify the directory exists
    if [ ! -d "$LONGFORMER_EMBEDDING_DIR_REL" ]; then
        echo "ERROR: Checkpoint directory exists but contains problematic characters: $LONGFORMER_EMBEDDING_DIR_REL" | tee -a "$LOG_DIR/16_train_long_embedding.log"
        exit 1
    fi
    
    LONGFORMER_EMBEDDING_CKPT_NAME=$(basename "$LONGFORMER_EMBEDDING_DIR_REL")
    # Sanitize the checkpoint name too
    LONGFORMER_EMBEDDING_CKPT_NAME=$(sanitize_path "$LONGFORMER_EMBEDDING_CKPT_NAME")
    
    echo "Found Longformer Embedding checkpoint from previous run: $LONGFORMER_EMBEDDING_DIR_REL (Name: $LONGFORMER_EMBEDDING_CKPT_NAME)" | tee -a "$LOG_DIR/16_train_long_embedding.log"
fi

# 17. Run Inference (Longformer Embedding)
if should_run_step 17; then
    echo "[Step 17/$TOTAL_STEPS] Running Inference (Longformer Embedding)..." | tee "$LOG_DIR/17_infer_long_embedding.log"
    log_exec "$LOG_DIR/17_infer_long_embedding.log" python inference.py --checkpoint "$LONGFORMER_EMBEDDING_CKPT_NAME" --model long --batch $INFER_BATCH_SIZE --mask 0.75
    echo "Inference (Longformer Embedding) finished."

    # Delete .npy files after inference
    delete_npy_files "$LONGFORMER_EMBEDDING_DIR_REL" "$LOG_DIR/17_infer_long_embedding.log"

    # Archive and send inference log
    echo "Archiving and sending Longformer Embedding inference log..." | tee -a "$LOG_DIR/17_infer_long_embedding.log"
    create_archive "Long_Embedding_Inference_Log" "$LOG_DIR/17_infer_long_embedding.log" "Longformer Embedding Inference Log" "$LOG_DIR/17_infer_long_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Embedding_Inference_Log.zip" "Longformer Embedding Inference Log" "$LOG_DIR/17_infer_long_embedding.log"
else
    skip_step 17 "Running Inference (Longformer Embedding)" "$LOG_DIR/17_infer_long_embedding.log"
fi

# 18. Run Integrated Gradients (Longformer Embedding) + Archive and Send
if should_run_step 18; then
    echo "[Step 18/$TOTAL_STEPS] Running Integrated Gradients (Longformer Embedding)..." | tee "$LOG_DIR/18_intgrad_long_embedding.log"
    echo "Starting integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/18_intgrad_long_embedding.log"
    log_exec "$LOG_DIR/18_intgrad_long_embedding.log" python visualize/int_grad.py --checkpoint "$LONGFORMER_EMBEDDING_CKPT_NAME" --model long --CF --pre --n_steps 20
    echo "Completed integrated gradients calculation at $(date)" | tee -a "$LOG_DIR/18_intgrad_long_embedding.log"
    echo "Integrated Gradients (Longformer Embedding) finished."

    # Delete .npy files after integrated gradients
    delete_npy_files "$LONGFORMER_EMBEDDING_DIR_REL" "$LOG_DIR/18_intgrad_long_embedding.log"

    # Create archive of IntGrad results and checkpoint
    echo "[Step 18/$TOTAL_STEPS] Archiving and Sending Longformer Embedding Results..." | tee -a "$LOG_DIR/18_intgrad_long_embedding.log"
    create_archive "Long_Emb_IntGrad" "PATTERN:$LONGFORMER_EMBEDDING_DIR_REL:*.png *.npy" "Longformer Embedding IntGrad Results" "$LOG_DIR/18_intgrad_long_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Emb_IntGrad.zip" "Longformer Embedding IntGrad Results" "$LOG_DIR/18_intgrad_long_embedding.log"
    create_archive "Long_Emb_Checkpoint" "$LONGFORMER_EMBEDDING_DIR_REL" "Longformer Embedding Checkpoint" "$LOG_DIR/18_intgrad_long_embedding.log"
    send_with_runpodctl "${ARCHIVE_DIR}/Long_Emb_Checkpoint.zip" "Longformer Embedding Checkpoint" "$LOG_DIR/18_intgrad_long_embedding.log"
else
    skip_step 18 "Running Integrated Gradients (Longformer Embedding)" "$LOG_DIR/18_intgrad_long_embedding.log"
fi

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