import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# ecog_data is the output of extract_neural_data, freq_range is a tuple (a, b), where a < b
def psd_welch(ecog_data, lower_freq, upper_freq, n_per_seg=128):
    assert lower_freq < upper_freq, 'lower_freq must be less than upper_freq'
    signal = ecog_data['signal']
    psd, freqs = psd_array_welch(signal, sfreq=ecog_data['sampling_rate'], n_per_seg=n_per_seg)
    freq_cap = (freqs >= lower_freq) & (freqs <= upper_freq)

    psd = psd[freq_cap]
    freqs = freqs[freq_cap]

    channel = ecog_data['channel']
    behavior = ecog_data['behavior'] if ecog_data['behavior'] is not None else 'all states'
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, 10 * np.log10(psd))  # Convert to dB
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(f'PSD (Welch) - Averaged Signal, Channel {channel}, {behavior}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Specify nperseg as a power of 2, lower value = high time res., low freq. res.
# higher value = low time res., high freq. res.
def plot_spectrogram(ecog_data, lower_freq, upper_freq, nperseg=256, baseline_correction='z-score'):
    event_epochs = ecog_data['signal']
    pre_time, post_time = ecog_data['window']
    signal = ecog_data['signal']
    sampling_rate = ecog_data['sampling_rate']
    channel = ecog_data['channel']
    behavior = ecog_data['behavior'] if ecog_data['behavior'] is not None else 'all states'

    Sxx_epochs = []

    for i in range(event_epochs.shape[0]):
        signal = event_epochs[i, :, channel]
        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2)
        t -= pre_time
        freq_cap = (f >= lower_freq) & (f <= upper_freq)
        f = f[freq_cap]
        Sxx = Sxx[freq_cap, :]

        baseline_mask = (t >= -pre_time) & (t <= -pre_time / 2)  # the interval we use to calculate baseline
        baseline_power = Sxx[:, baseline_mask].mean(axis=1, keepdims=True)

        # z-score
        mu = baseline_power
        sig = Sxx[:, baseline_mask].std(axis=1, keepdims=True)
        Sxx_z = (Sxx - mu) / sig

        Sxx_epochs.append(Sxx_z)

    Sxx_epochs = np.array(Sxx_epochs)
    Sxx = np.mean(Sxx_epochs, axis=0)
    
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram - Averaged over epochs, Channel {channel}, {behavior}, Baseline adjusted with z-score')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.show()
