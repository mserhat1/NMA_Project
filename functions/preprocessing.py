import numpy as np

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
