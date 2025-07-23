import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from scipy.signal import butter, hilbert,filtfilt,sosfiltfilt
from scipy.fft import rfft, rfftfreq


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


def extract_gammaH_LFO(ecog_data, gamma_band=(70, 150), lfo_band=(0.5, 4), time_window=None, zscore=True, clip_percentile=99.9):
    """
    To extract LFO of high-gamma envelope
    Reference: Natraj et al., 2022, Neuron 

    """
    signal = ecog_data['signal']
    fs = ecog_data['sampling_rate']

    # time window
    if time_window is not None and time_window != 'all':
        t1, t2 = time_window
        s1, s2 = int(t1 * fs), int(t2 * fs)
        signal = signal[s1:s2]

    # shape
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    # define filter parameter
    # b_gamma, a_gamma = butter(4, [gamma_band[0]/(fs/2), gamma_band[1]/(fs/2)], btype='band')
    # b_lfo, a_lfo = butter(4, [lfo_band[0]/(fs/2), lfo_band[1]/(fs/2)], btype='band')

    # attempt: sos filter parameter
    # filtfilt works fine with gamma bandpass filter but will lose information (all values would be 0 for LFO filter, so changed to sosfiltfilt)
    sos_gamma = butter(4, [gamma_band[0]/(fs/2), gamma_band[1]/(fs/2)], btype='band', output='sos')
    sos_lfo   = butter(4, [lfo_band[0]/(fs/2), lfo_band[1]/(fs/2)], btype='band', output='sos')

    gamma_lfo = []
    gamma_nolfo = []
    for ch in signal:
        # high gamma filter
        # gamma = filtfilt(b_gamma, a_gamma, ch)
        gamma = sosfiltfilt(sos_gamma, ch)
        # Hilbert envelope
        gamma_env = np.abs(hilbert(gamma))
        # clip: prevent abnormal values
        gamma_env = np.clip(gamma_env, 0, np.percentile(gamma_env, clip_percentile))
        # LFO
        # gamma_env_lfo = filtfilt(b_lfo, a_lfo, gamma_env)
        gamma_env_lfo = sosfiltfilt(sos_lfo, gamma_env)
        # normalization?
        if zscore:
            # gamma_env = (gamma_env - np.mean(gamma_env)) / np.std(gamma_env)
            gamma_env_lfo = (gamma_env_lfo - np.mean(gamma_env_lfo)) / np.std(gamma_env_lfo)

        gamma_lfo.append(gamma_env_lfo)
        # gamma_nolfo.append(gamma_env)

    return np.array(gamma_lfo)


def extract_LFO_envelope(ecog_data, lfo_band=(0.5, 4), time_window=None, zscore=True):
    """
    Extract low-frequency oscillation (LFO) envelope from raw ECoG signal.
    Reference: Natraj et al., 2022, Neuron 
    """
    signal = ecog_data['signal']
    fs = ecog_data['sampling_rate']

    # time window
    if time_window is not None and time_window != 'all':
        t1, t2 = time_window
        s1, s2 = int(t1 * fs), int(t2 * fs)
        signal = signal[s1:s2]

    # shape to (channels, time)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.shape[0] > signal.shape[1]:
        signal = signal.T

    # bandpass filter (4th-order IIR)
    # b_lfo, a_lfo = butter(4, [lfo_band[0]/(fs/2), lfo_band[1]/(fs/2)], btype='band')
    sos_lfo   = butter(4, [lfo_band[0]/(fs/2), lfo_band[1]/(fs/2)], btype='band', output='sos')

    lfo_envelopes = []
    for ch in signal:
        # lfo = filtfilt(b_lfo, a_lfo, ch)
        lfo = sosfiltfilt(sos_lfo, ch)
        env = np.abs(hilbert(lfo))  # analytic amplitude
        if zscore:
            env = (env - np.mean(env)) / np.std(env)
        lfo_envelopes.append(env)

    return np.array(lfo_envelopes)  # shape: (channels, time)


def plot_gamma_lfo(gamma_lfo, fs, channel=0, title_prefix='High-gamma LFO envelope'):

    time_axis = np.arange(gamma_lfo.shape[1]) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, gamma_lfo[channel])
    plt.title(f'{title_prefix} (channel {channel})')
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope amplitude")
    plt.tight_layout()
    plt.show()


def plot_lfo_envelope(lfo_envelope, fs, channel=0, title_prefix='LFO envelope'):
    time_axis = np.arange(lfo_envelope.shape[1]) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, lfo_envelope[channel])
    plt.title(f'{title_prefix} (channel {channel})')
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope amplitude")
    plt.tight_layout()
    plt.show()

def plot_power_spectrum(ecog_data, channel=0, f_max=20):
    """
    check power spectrum for raw EcoG data to confirm there are signals in low frequency band
    """
    signal = ecog_data['signal']
    fs = ecog_data['sampling_rate']

    x = signal[:, channel] if signal.shape[1] > signal.shape[0] else signal[channel]

    f = rfftfreq(len(x), 1 / fs)
    spec = np.abs(rfft(x))

    plt.figure(figsize=(10, 4))
    plt.plot(f, spec)
    plt.xlim(0, f_max)
    plt.title(f"ECoG Power Spectrum (channel {channel})")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
