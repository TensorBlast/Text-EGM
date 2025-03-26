def z_score_normalization(data):
    mean_val = np.mean(data, axis=(0, 1), keepdims=True)
    std_val = np.std(data, axis=(0, 1), keepdims=True)
    # Avoid division by zero
    std_val = np.maximum(std_val, 1e-10)
    normalized_data = (data - mean_val) / std_val
    # Ensure no NaNs in the output
    normalized_data = np.nan_to_num(normalized_data, nan=0.0)
    return normalized_data 