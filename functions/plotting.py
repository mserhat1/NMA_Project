import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import pywt

import matplotlib.pyplot as plt
import numpy as np

def plot_spatial_data(spatial_data, keypoints, time_window='all', plot_histogram=True):
    for kp in keypoints:
        if time_window == 'all':
            timestamps = spatial_data[kp]['timestamps']
            x_axis = spatial_data[kp]['data'][:, 0]
            y_axis = spatial_data[kp]['data'][:, 1]
        else:
            rate = spatial_data[kp]['sampling_rate']
            t1, t2 = time_window
            assert t1 >= 0 and t2 <= spatial_data[kp]['data'].shape[0]/rate and t1 < t2, 'We need 0 <= t1 < t2 <= duration of data'
            s1 = int(t1 * rate)
            s2 = int(t2 * rate)
            timestamps = spatial_data[kp]['timestamps'][s1:s2]
            x_axis = spatial_data[kp]['data'][s1:s2, 0]
            y_axis = spatial_data[kp]['data'][s1:s2, 1]

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, x_axis, label=kp + ' X')
        plt.plot(timestamps, y_axis, label=kp + ' Y')

        plt.xlabel('Time (s)')
        plt.ylabel('Position (pixels)')
        plt.title('Time Course of ' + kp + ' Position')
        plt.legend()
        plt.grid(True)
        plt.show()

        if plot_histogram:
            # Plotting histograms
            plt.figure(figsize=(12, 6))
            plt.hist([x_axis, y_axis], label=['x axis', 'y axis'], bins='auto', edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.show()

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
        signal = event_epochs[i, :]
        if np.count_nonzero(np.isnan(signal)) > 0:
          continue
        f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 8)
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
    Sxx_avg = np.mean(Sxx_epochs, axis=0)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx_avg, shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram - Averaged over epochs, Channel {channel}, {behavior}, Baseline adjusted with z-score')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.show()

# applies z-score baseline correction per epoch, then averages the transform
def plot_morlet(ecog_data, lower_freq, upper_freq, B, C):
    event_epochs = ecog_data['signal']
    pre_time, post_time = ecog_data['window']
    dt = 1 / ecog_data['sampling_rate']
    freqs = np.arange(lower_freq, upper_freq + 2, 2)
    scales = pywt.frequency2scale(f'cmor{B}-{C}', freqs * dt)

    wt_epochs = []

    t = np.linspace(-pre_time, post_time, len(event_epochs[0]))
    b_mask = (t >= -pre_time) & (t <= -pre_time / 2)

    z_sum = 0
    n_count = 0

    for i in range(event_epochs.shape[0]):
        signal = event_epochs[i, :]
        if np.count_nonzero(np.isnan(signal)) > 0:
            continue
        cwt, _ = pywt.cwt(signal, scales, f'cmor{B}-{C}', sampling_period=dt)
        power = np.abs(cwt) ** 2

        mu  = power[:, b_mask].mean(axis=1, keepdims=True)
        sigma  = power[:, b_mask].std(axis=1,  keepdims=True) + 1e-10
        z  = (power - mu) / sigma

        z_sum += z
        n_count += 1

    z_avg = z_sum / n_count

    plt.figure(figsize=(10, 6))
    plt.imshow(z_avg, extent=[t[0], t[-1], freqs[0], freqs[-1]],
               cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Power')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Morlet Scalogram')
    plt.tight_layout()
    plt.show()
