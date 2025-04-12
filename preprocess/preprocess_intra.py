import os
import wfdb
import numpy as np
import argparse
import requests
import zipfile
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path', type=str, default=None, 
                       help='Path to the folder containing the data. If not provided, data will be downloaded automatically.')
    return parser.parse_args()

def download_and_extract_data(data_dir):
    """
    Downloads and extracts the intracardiac AF database if not already present.
    
    Args:
        data_dir: Directory where the data should be downloaded and extracted to
    
    Returns:
        Path to the extracted data directory
    """
    zip_path = os.path.join(data_dir, "intracardiac-atrial-fibrillation-database-1.0.0.zip")
    extract_dir = os.path.join(data_dir, "intracardiac-atrial-fibrillation-database-1.0.0")
    url = "https://physionet.org/static/published-projects/iafdb/intracardiac-atrial-fibrillation-database-1.0.0.zip"
    
    # Check if data is already extracted
    if os.path.exists(extract_dir):
        print(f"Data already exists at {extract_dir}")
        return extract_dir
        
    # Create data directory if it doesn't exist
    ensure_directory_exists(data_dir)
    
    # Flag to track if zip file existed before download attempt
    zip_existed_before = os.path.exists(zip_path)
    
    # Download if zip file doesn't exist
    if not zip_existed_before:
        download_successful = download_file(url, zip_path)
        if not download_successful:
            return None
    else:
        print(f"Zip file already exists at {zip_path}")
    
    # Try to extract the zip file
    extract_successful = False
    try:
        print(f"Extracting data to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction completed successfully!")
        extract_successful = True
    except Exception as e:
        print(f"Error extracting data: {e}")
        # If extraction failed and zip existed before, try to redownload
        if zip_existed_before:
            print("Zip file may be corrupted. Will redownload and try again.")
            # Remove the corrupted zip file
            try:
                os.remove(zip_path)
                print(f"Removed corrupted zip file: {zip_path}")
            except Exception as rm_err:
                print(f"Warning: Failed to remove corrupted zip file: {rm_err}")
            
            # Redownload
            download_successful = download_file(url, zip_path)
            if download_successful:
                # Try extraction again
                try:
                    print(f"Trying extraction again after redownload...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print("Extraction completed successfully after redownload!")
                    extract_successful = True
                except Exception as retry_err:
                    print(f"Error extracting data after redownload: {retry_err}")
    
    if extract_successful:
        return extract_dir
    else:
        return None

def download_file(url, zip_path):
    """
    Downloads a file from the given URL to the specified path.
    
    Args:
        url: URL to download from
        zip_path: Local path to save the file to
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Try using wget first (usually faster)
    try:
        print(f"Downloading data using wget to {zip_path}...")
        result = subprocess.run(
            ["wget", "-O", zip_path, url, "--progress=bar:force:noscroll"], 
            check=True
        )
        if result.returncode == 0:
            print("Download completed successfully!")
            return True
        else:
            raise Exception("wget command failed")
    except Exception as e:
        print(f"wget download failed: {e}")
        print("Falling back to Python requests for download...")
        
        # If wget fails, fall back to requests with a progress bar
        try:
            print(f"Downloading data to {zip_path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get the file size from headers
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Show download progress bar
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            
            with open(zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong with the download")
                return False
            
            print("Download completed successfully!")
            return True
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False

# def z_score_normalization(data):
#     mean_val = np.mean(data, axis=(0, 1), keepdims=True)
#     std_val = np.std(data, axis=(0, 1), keepdims=True)
#     normalized_data = (data - mean_val) / std_val
#     return normalized_data

def z_score_normalization(data):
    mean_val = np.mean(data, axis=(0, 1), keepdims=True)
    std_val = np.std(data, axis=(0, 1), keepdims=True)
    # Avoid division by zero
    std_val = np.maximum(std_val, 1e-10)
    normalized_data = (data - mean_val) / std_val
    # Ensure no NaNs in the output
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    return normalized_data

def segment_signal(data, segment_length, step_size = None):
    
    n_time_points, n_electrodes, n_placements = data.shape
    
    if step_size != None:
        n_segments = 1 + (n_time_points - segment_length) // step_size
        segmented_data = np.zeros((n_segments, segment_length, n_electrodes, n_placements))

        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_length
            segmented_data[i] = data[start_idx:end_idx, :, :]
            
    elif step_size == None:
        n_segments = n_time_points // segment_length
        truncated_data = data[:n_segments * segment_length]
        segmented_data = truncated_data.reshape(n_segments, segment_length, data.shape[1], data.shape[2])
    
    return segmented_data

def read_all(path):
    print(f"Looking for files in: {path}")
    all_files = os.listdir(path)
    print(f"Found {len(all_files)} files")
    print(f"Sample filenames: {all_files[:5] if len(all_files) > 5 else all_files}")
    
    qrs_files = [file for file in all_files if 'qrs' in file]
    print(f"Found {len(qrs_files)} files with 'qrs' in the name")
    
    # Rest of your original function
    all_signals = []
    for i in qrs_files:
        file_name = i.split('.')[0]
        try:
            record = wfdb.rdrecord(os.path.join(path, file_name))
            egm_signals = record.p_signal[:, 3:]
            all_signals.append(egm_signals)
            print(f"Successfully processed {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    if not all_signals:
        print("No signals were processed!")
        return None
        
    min_shape = min(array.shape[0] for array in all_signals)
    sliced_arrays = [array[:min_shape] for array in all_signals]
    stacked_array = np.stack(sliced_arrays, axis=-1)
    
    return stacked_array

def split_dict_by_catheter_afib(input_dict):

    train_dict, test_dict, val_dict = {}, {}, {}
    num_catheters = len(set([key[1] for key in input_dict.keys()]))
    for key, value in input_dict.items():
        _, catheter_num, _, _ = key
        if catheter_num < 21:
            train_dict[key] = value
        elif 21 <= catheter_num < 26:
            test_dict[key] = value
        elif 26 <= catheter_num:
            val_dict[key] = value
    print(f"Number of catheters: {num_catheters}")
    return train_dict, test_dict, val_dict

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def main(args):
    # Handle path argument
    if args.path is None:
        # Use default path and download if necessary
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        print(f"No path provided. Will use default data directory: {data_dir}")
        data_path = download_and_extract_data(data_dir)
        if data_path is None:
            print("Failed to download or extract data. Exiting.")
            return
        args.path = data_path
    else:
        # Verify the provided path exists
        if not os.path.exists(args.path):
            print(f"Error: The provided path '{args.path}' does not exist. Exiting.")
            return
        print(f"Using provided path: {args.path}")
    
    print("Starting data processing...")
    egm_signals = read_all(args.path)
    if egm_signals is None:
        print("No signals to process. Exiting.")
        return
    
    print(f"Normalizing data of shape {egm_signals.shape}...")
    normalized_egm = z_score_normalization(egm_signals)
    
    print(f"Segmenting data...")
    segmented_data = segment_signal(normalized_egm, 1000)
    print(f"Segmented data shape: {segmented_data.shape}")

    print("Creating dictionary...")
    segmented_data_dict = {}
    n_segments, segment_length, n_electrodes, n_placements = segmented_data.shape
    for i in range(n_electrodes):
        for j in range(n_placements):
            for k in range(n_segments):
                key = (i, j, k, 1)
                segmented_data_dict[key] = segmented_data[k, :, i, j]
    print(f"Dictionary contains {len(segmented_data_dict)} entries")

    feature_dicts = {
            'segmented_data': segmented_data_dict,
            }
    
    print("Concatenating features...")
    concatenated_features = {}
    for key in feature_dicts['segmented_data'].keys():
        concatenated_feature = np.concatenate([feature_dicts[feature_name][key] for feature_name in feature_dicts])        
        concatenated_features[key] = concatenated_feature
    
    print("Splitting data...")
    train, test, val = split_dict_by_catheter_afib(concatenated_features)
    print(f"Train: {len(train)} entries, Test: {len(test)} entries, Val: {len(val)} entries")

    ensure_directory_exists('../data')

    print("Saving data...")
    np.save('../data/train_intra.npy', train)
    np.save('../data/val_intra.npy', val)
    np.save('../data/test_intra.npy', test)
    print("Data saved successfully!")

if __name__ == '__main__':
    args = get_args()
    main(args)
