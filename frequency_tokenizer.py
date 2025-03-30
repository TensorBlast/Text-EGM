import numpy as np
from scipy import signal


class FrequencyTokenizer:
    """Tokenizes the frequency components of ECG signals"""
    def __init__(self, n_freq_bins=50, signal_size=250, sampling_rate=1000):
        """
        Initialize the frequency tokenizer
        
        Args:
            n_freq_bins: Number of frequency bins to use
            signal_size: Max number of discrete values for quantization
            sampling_rate: Sampling rate of the ECG signal
        """
        self.n_freq_bins = n_freq_bins
        self.signal_size = signal_size
        self.sampling_rate = sampling_rate
        
    def compute_stft(self, ecg_signal, window_length=128, hop_length=64):
        """
        Compute Short-Time Fourier Transform of the signal
        
        Args:
            ecg_signal: ECG signal array
            window_length: STFT window length
            hop_length: STFT hop length
            
        Returns:
            Time-frequency representation of the signal
        """
        f, t, Zxx = signal.stft(ecg_signal, fs=self.sampling_rate, 
                                nperseg=window_length, 
                                noverlap=window_length-hop_length)
        # Take magnitude of complex values
        Zxx = np.abs(Zxx)
        # Limit to lower frequency range where most ECG information exists
        max_freq_idx = min(len(f), self.n_freq_bins)
        Zxx = Zxx[:max_freq_idx, :]
        return Zxx, f[:max_freq_idx], t
    
    def tokenize(self, ecg_signal):
        """
        Convert signal to frequency tokens
        
        Args:
            ecg_signal: ECG signal array
            
        Returns:
            Frequency tokens, min value, max value
        """
        # Compute STFT
        spectrogram, freqs, times = self.compute_stft(ecg_signal)
        
        # Normalize spectrogram
        spec_min, spec_max = np.min(spectrogram), np.max(spectrogram)
        if spec_max > spec_min:  # Avoid division by zero
            norm_spec = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            norm_spec = np.zeros_like(spectrogram)
            
        # Quantize to discrete tokens
        quantized_spec = np.floor(norm_spec * self.signal_size).astype(int)
        
        # Convert to token format
        freq_tokens = []
        for i in range(quantized_spec.shape[0]):
            for j in range(quantized_spec.shape[1]):
                freq_tokens.append(f"freq_{i}_{quantized_spec[i, j]}")
                
        return freq_tokens, spec_min, spec_max, quantized_spec, freqs, times
    
    def get_spec_shape(self, ecg_signal, window_length=128, hop_length=64):
        """
        Get the shape of the spectrogram for a given signal
        
        Args:
            ecg_signal: ECG signal array
            window_length: STFT window length
            hop_length: STFT hop length
            
        Returns:
            Shape of the spectrogram
        """
        spectrogram, _, _ = self.compute_stft(ecg_signal, window_length, hop_length)
        return spectrogram.shape 