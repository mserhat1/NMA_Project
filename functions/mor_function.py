import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

def extract_wavelet_transform(ecog_data, time_window=None, wavelet='cmor1.5-1.0', 
                            freq_range=(1, 150), num_freqs=100):
    """
    Calculate continuous wavelet transform (CWT)
    
    Parameters:
    - ecog_data: Dictionary containing 'signal' and 'sampling_rate'
    - time_window: Time window (t1, t2) or 'all' or None
    - wavelet: Wavelet type, default 'cmor1.5-1.0' (Complex Morlet wavelet)
    - freq_range: Frequency range (min_freq, max_freq)
    - num_freqs: Number of frequency points
    
    Returns:
    - cwt_array: Wavelet transform result array (channels, frequencies, time)
    - frequencies: Frequency array
    - times: Time array
    """
    signal_data = ecog_data['signal'].copy()  # Use copy to avoid modifying original data
    fs = ecog_data['sampling_rate']
    
    print(f"Original signal shape: {signal_data.shape}")
    
    # Ensure correct signal format - following your spectrogram function logic
    if signal_data.ndim == 1:
        signal_data = signal_data[np.newaxis, :]
        print(f"1D signal converted shape: {signal_data.shape}")
    
    # Process dimensions following your spectrogram function logic
    if signal_data.shape[0] < signal_data.shape[1]:
        signal_data = signal_data  # Keep as is (channels, samples)
    else:
        signal_data = signal_data.T  # Transpose to (channels, samples)
    
    print(f"Shape after dimension processing: {signal_data.shape}")
    
    # Time window processing - slice along time dimension
    if time_window != 'all' and time_window is not None:
        t1, t2 = time_window
        s1, s2 = int(t1 * fs), int(t2 * fs)
        # Ensure valid indices
        s1 = max(0, s1)
        s2 = min(signal_data.shape[1], s2)
        if s1 >= s2:
            raise ValueError(f"Invalid time window: [{t1}, {t2}]s corresponds to samples[{s1}, {s2}]")
        signal_data = signal_data[:, s1:s2]
        print(f"Shape after time window slicing: {signal_data.shape}")
    
    # Check if data is empty
    if signal_data.size == 0:
        raise ValueError("Signal data is empty, please check input data and time window settings")
    
    print(f"Final processed signal shape: {signal_data.shape}")
    
    # Generate frequency scales
    min_freq, max_freq = freq_range
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), num_freqs)
    
    # Convert frequencies to scales - fix scales calculation
    if 'cmor' in wavelet:
        # For complex Morlet wavelet, use more stable scale calculation method
        # cmor1.5-1.0 means bandwidth=1.5, center_frequency=1.0
        parts = wavelet.split('-')
        if len(parts) == 2:
            center_freq = float(parts[1])
        else:
            center_freq = 1.0
        
        # Use PyWavelets recommended method to calculate scales
        scales = center_freq * fs / frequencies
        # Ensure scales are positive and reasonable
        scales = np.clip(scales, a_min=1, a_max=None)
        
    else:
        # For other wavelet types
        scales = pywt.scale2frequency(wavelet, 1/frequencies) * fs
        scales = np.clip(scales, a_min=1, a_max=None)
    
    print(f"Frequency range: {min_freq}-{max_freq} Hz")
    print(f"Scale range: {np.min(scales):.2f}-{np.max(scales):.2f}")
    
    # Check validity of scales array
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        print(f"Warning: scales array contains invalid values")
        print(f"scales: {scales}")
        # Remove invalid values
        valid_mask = np.isfinite(scales) & (scales > 0)
        scales = scales[valid_mask]
        frequencies = frequencies[valid_mask]
        if len(scales) == 0:
            raise ValueError("No valid scale values, please check frequency range and sampling rate settings")
    
    # Generate time axis
    times = np.arange(signal_data.shape[1]) / fs
    
    cwt_list = []
    
    print(f"Starting to process {signal_data.shape[0]} channels, signal length: {signal_data.shape[1]}")
    
    for i, ch in enumerate(signal_data):
        if i % 20 == 0:  # Print progress every 20 channels
            print(f"Processing channel {i}/{signal_data.shape[0]}")
        
        # Check channel data
        if len(ch) == 0:
            raise ValueError(f"Channel {i} data is empty")
        
        # For very long signals, may need preprocessing or segmented processing
        if len(ch) > 100000:  # If signal is too long, give a hint
            if i == 0:  # Only hint on first channel
                print(f"Signal is long ({len(ch)} samples), wavelet transform may take some time...")
        
        try:
            # Calculate continuous wavelet transform
            cwt_matrix, _ = pywt.cwt(ch, scales, wavelet, sampling_period=1/fs)
            
            # Calculate power spectrum (square of magnitude, then log)
            cwt_power = np.abs(cwt_matrix) ** 2
            cwt_log = np.log1p(cwt_power)
            
            cwt_list.append(cwt_log)
            
        except Exception as e:
            print(f"Channel {i} processing failed: {e}")
            print(f"Channel {i} data statistics: min={np.min(ch):.3f}, max={np.max(ch):.3f}, mean={np.mean(ch):.3f}")
            print(f"Data type: {ch.dtype}")
            print(f"Contains NaN: {np.any(np.isnan(ch))}")
            print(f"Contains Inf: {np.any(np.isinf(ch))}")
            raise
    
    result = np.stack(cwt_list)
    print(f"Wavelet transform completed! Final result shape: {result.shape}")
    
    return result, frequencies, times


def plot_wavelet_channel(cwt_array, frequencies, times, ch_index=0, freq_max=150):
    """
    Plot wavelet transform results, y-axis: 0–freq_max Hz
    
    Parameters:
    - cwt_array: Wavelet transform result array
    - frequencies: Frequency array
    - times: Time array
    - ch_index: Channel index to plot (default 0)
    - freq_max: Maximum frequency to display (default 150)
    """
    # Check if channel index is valid
    if ch_index >= cwt_array.shape[0]:
        raise ValueError(f"Channel index {ch_index} out of range, data has only {cwt_array.shape[0]} channels (indices 0-{cwt_array.shape[0]-1})")
    
    # Debug information
    print(f"Plot data check:")
    print(f"cwt_array shape: {cwt_array.shape}")
    print(f"frequencies shape: {frequencies.shape}, range: {frequencies.min():.2f}-{frequencies.max():.2f}")
    print(f"times shape: {times.shape}, range: {times.min():.2f}-{times.max():.2f}")
    
    freq_mask = frequencies <= freq_max
    f_plot = frequencies[freq_mask]
    cwt_plot = cwt_array[ch_index][freq_mask, :]
    
    print(f"Plot data statistics:")
    print(f"f_plot range: {f_plot.min():.2f}-{f_plot.max():.2f}")
    print(f"cwt_plot shape: {cwt_plot.shape}")
    print(f"cwt_plot value range: {cwt_plot.min():.3f}-{cwt_plot.max():.3f}")
    print(f"cwt_plot all zeros: {np.all(cwt_plot == 0)}")
    
    plt.figure(figsize=(12, 6))
    
    # Use more stable plotting method
    im = plt.imshow(cwt_plot, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], f_plot[0], f_plot[-1]],
                    cmap='viridis')
    
    plt.colorbar(im, label='Log Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Wavelet Transform (channel {ch_index})')
    
    # Set y-axis range and ticks
    plt.ylim(f_plot[0], min(freq_max, f_plot[-1]))
    
    # Use linear axis first, change to log axis if needed
    # plt.yscale('log')  # Temporarily commented out log axis
    
    plt.tight_layout()
    plt.show()


# Usage example
"""

# Calculate wavelet transform
cwt_result, freqs, times = extract_wavelet_transform(
    ecog_data, 
    time_window=(0, 30),  # Process first 30 seconds, or use None for all
    wavelet='cmor1.5-1.0',  # Complex Morlet wavelet
    freq_range=(1, 150),
    num_freqs=100
)

# Plot wavelet transform for specified channel
plot_wavelet_channel(cwt_result, freqs, times, ch_index=0, freq_max=150)
"""
