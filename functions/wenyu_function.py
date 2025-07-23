import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


def extract_spectrogram(ecog_data, time_window=None, nperseg=256, noverlap=128):
    """
    calculate spectrogram（log-scale）
    """
    signal = ecog_data['signal']
    fs = ecog_data['sampling_rate']

    if time_window != 'all' and time_window is not None:
        t1, t2 = time_window
        s1, s2 = int(t1 * fs), int(t2 * fs)
        signal = signal[s1:s2]

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    if signal.shape[0] < signal.shape[1]:
        signal = signal
    else:
        signal = signal.T

    spec_list = []
    f, t = None, None
    for ch in signal:
        f, t, Sxx = spectrogram(ch, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx_log = np.log1p(Sxx)
        spec_list.append(Sxx_log)

    return np.stack(spec_list), f, t


def plot_spectrogram_channel(spec_array, f_spec, t_spec, ch_index=0, freq_max=150):
    """
    plot spectrogram，y axis: 0–freq_max Hz

    parameters:
    - ch_index: channel to plot (default 0)
    - freq_max: max frequency to show (default 150)
    """
    freq_mask = f_spec <= freq_max
    f_plot = f_spec[freq_mask]
    spec_plot = spec_array[ch_index][freq_mask, :]

    plt.figure(figsize=(10, 5))
    plt.imshow(spec_plot, aspect='auto', origin='lower',
               extent=[t_spec[0], t_spec[-1], f_plot[0], f_plot[-1]])
    plt.colorbar(label='Log Power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Spectrogram (channel {ch_index})')
    plt.ylim(0, freq_max)
    plt.tight_layout()
    plt.show()