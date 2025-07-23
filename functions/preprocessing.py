import numpy as np
from scipy.signal import butter, filtfilt

# can use the output of extract_neural_data as the input to avg_across_epochs
def avg_across_epochs(ecog_data):
    signal = ecog_data['signal']
    average_signal = np.nanmean(signal, axis=0) # shape = (n_samples, )
    ecog_data['signal'] = average_signal
    return ecog_data

def ptp_thresholding(ecog_data, threshold_uv=200):
    event_epochs = ecog_data['signal']

    ptp = np.ptp(event_epochs, axis=1)
    ptp_mask = ptp <= threshold_uv
    new_epochs = event_epochs[ptp_mask]

    filtered_data = ecog_data.copy()
    filtered_data['signal'] = new_epochs
    print(f"Kept {len(new_epochs)} out of {len(event_epochs)} epochs after amplitude filtering.")

    return filtered_data

# available modes are 'zero' and 'interp'
def spike_cleaning(ecog_data, z_thresh=5.0, mode='zero', return_spike_mask=False):
    signal = ecog_data['signal']
    cleaned_signal = signal.copy()

    total_spikes = 0
    spike_mask_all = np.zeros_like(signal, dtype=bool)

    z = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    spike_mask = np.abs(z) > z_thresh
    num_spikes = np.sum(spike_mask)

    if mode == 'zero':
        signal[spike_mask] = 0.0
    elif mode == 'interp':
        signal[spike_mask] = np.interp(np.flatnonzero(spike_mask),
                                          np.flatnonzero(~spike_mask),
                                          signal[~spike_mask])
    else:
        raise ValueError("mod must be 'zero' or 'interp'")

    cleaned_data = ecog_data.copy()
    cleaned_data['signal'] = signal

    print(f"Spike cleaning applied: z > {z_thresh} → mode: {mode}")
    print(f"Total spikes removed/interpolated: {num_spikes}")

    if return_spike_mask:
        return cleaned_data, spike_mask
    else:
        return cleaned_data
        
def bandpass_filter(ecog_data, lowcut=0.5, highcut=4.0, order=4):
    """
    Apply a bandpass filter to a multi-channel signal.
    
    Parameters:
        signal (np.ndarray): Input array of shape (n_samples, n_channels).
        fs (float): Sampling rate in Hz.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        order (int): Order of the Butterworth filter.
    
    Returns:
        np.ndarray: Bandpass-filtered signal of the same shape.
    """
    signal = ecog_data['signal']
    signal = signal.astype(np.float64)
    fs = ecog_data['sampling_rate']

    if signal.ndim != 2:
        raise ValueError("Signal must be a 2D array (n_samples, n_channels)")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    # Apply filter channel-wise
    filtered = np.empty_like(signal)
    for i in range(signal.shape[1]):
        filtered[:, i] = filtfilt(b, a, signal[:, i], axis=0)

    filtered_ecog = ecog_data.copy()
    filtered_ecog['signal'] = filtered.astype(np.float64)

    return filtered_ecog

