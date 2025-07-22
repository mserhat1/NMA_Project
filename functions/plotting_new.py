import numpy as np
from mne.time_frequency import psd_array_welch
from scipy.signal import spectrogram
import pywt
import matplotlib.pyplot as plt

def plot_signal(ecog_data):
    signal = ecog_data['signal']
    ch = ecog_data['channel']
    rate = ecog_data['sampling_rate']
    t1, t2 = ecog_data['window']
    timestamps = np.linspace(t1, t2, int((t2 - t1) * rate))

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (µV)')
    plt.title(f'ECoG Data, Channel {ch}')
    plt.grid(True)
    plt.show()

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
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
            # Plotting the histogram for x
            axes[0].hist(x_axis, label='x axis', bins='auto')
            axes[0].set_title('Histogram of ' + kp + ' X Position')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True)

            # Plotting the histogram for y
            axes[1].hist(y_axis, label='y axis', bins='auto', color='orange')
            axes[1].set_title('Histogram of ' + kp + ' Y Position')
            axes[1].set_xlabel('Value')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()


# ecog_data is the output of extract_neural_data, freq_range is a tuple (a, b), where a < b
def psd_welch(ecog_data, lower_freq, upper_freq, n_per_seg=128):
    assert lower_freq < upper_freq, 'lower_freq must be less than upper_freq'
    event_epochs = ecog_data['signal']
    psd, freqs = psd_array_welch(event_epochs, sfreq=ecog_data['sampling_rate'], n_per_seg=n_per_seg)
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
# Specify nperseg as a power of 2, lower value = high time res., low freq. res.
# higher value = low time res., high freq. res.
def plot_spectrogram(ecog_data, lower_freq, upper_freq, nperseg=256, baseline_correction=True):
    signal = ecog_data['signal']
    rate = ecog_data['sampling_rate']
    t1, t2 = ecog_data['window']

    # interpolating to clean nan's
    nans = np.isnan(signal)
    indices = np.arange(len(signal))
    signal_interp = np.interp(indices, indices[~nans], signal[~nans])

    f, t, Sxx = spectrogram(signal_interp, fs=ecog_data['sampling_rate'], nperseg=nperseg, noverlap=nperseg // 2)

    freq_cap = (f >= lower_freq) & (f <= upper_freq)
    f = f[freq_cap]
    Sxx = Sxx[freq_cap, :]

    if baseline_correction:
        baseline_mask = (t >= t1 ) & (t <= t1 + (t2 - t1) / 5)  # the interval we use to calculate baseline
        baseline_power = Sxx[:, baseline_mask].mean(axis=1, keepdims=True)

        # z-score
        mu = baseline_power
        sig = Sxx[:, baseline_mask].std(axis=1, keepdims=True)
        Sxx_final = (Sxx - mu) / sig
    else:
        Sxx_final = Sxx.copy()

    ch = ecog_data['channel']

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx_final, shading='gouraud')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f'Spectrogram - Averaged over epochs, Channel {ch}, Baseline adjusted with z-score')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.show()

def plot_morlet(ecog_data, lower_freq, upper_freq, B, C, baseline_correction=True):
    signal = ecog_data['signal']
    rate = ecog_data['sampling_rate']
    t1, t2 = ecog_data['window']
    #freqs = np.logspace(np.log10(lower_freq), np.log10(upper_freq), num=200) * dt
    freqs = np.arange(lower_freq, upper_freq + 2, 2)
    scales = pywt.frequency2scale(f'cmor{B}-{C}', freqs / rate)

    # interpolating to clean nan's
    nans = np.isnan(signal)
    indices = np.arange(len(signal))
    signal_interp = np.interp(indices, indices[~nans], signal[~nans])

    cwt, _ = pywt.cwt(signal_interp, scales, f'cmor{B}-{C}', sampling_period=1 / rate)
    power = np.abs(cwt) ** 2

    if baseline_correction:
        t = np.linspace(t1, t2, int((t2 - t1) * rate))
        b_mask = (t >= t1) & (t <= t1 + (t2 - t1) / 5)

        mu  = power[:, b_mask].mean(axis=1, keepdims=True)
        sigma  = power[:, b_mask].std(axis=1,  keepdims=True) + 1e-10
        z = (power - mu) / sigma


    plt.figure(figsize=(10, 6))
    plt.imshow(z, extent=[t[0], t[-1], freqs[0], freqs[-1]],
               cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Power')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Morlet Scalogram')
    plt.tight_layout()
    plt.show()
