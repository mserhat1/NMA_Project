from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
from ndx_events import LabeledEvents, AnnotatedEventsTable, Events
import fsspec
import h5py
from fsspec.implementations.cached import CachingFileSystem
import numpy as np

# Returns the file of the particular subject and the session;
def get_datafile(sbj, session):
    with DandiAPIClient() as client:
        asset = client.get_dandiset("000055").get_asset_by_path(
            "sub-{0:>02d}/sub-{0:>02d}_ses-{1:.0f}_behavior+ecephys.nwb".format(sbj, session)
        )
        s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
    fs = CachingFileSystem(
        fs=fsspec.filesystem("http"),
        cache_storage="nwb-cache",  # Local folder for the cache
    )

    f = fs.open(s3_path, "rb")
    file = h5py.File(f)
    io = NWBHDF5IO(file=file, mode='r', load_namespaces=True)
    nwbfile = io.read()

    return nwbfile

# behavior should be a list of activities whose neural data you are interested in.
# e.g. ['Talk', 'Eat', 'Computer/phone']
# if you want all the neural data in the file, do not specify behavior.
def extract_neural_data(nwbfile, channel, pre_time, post_time, behavior=None):
    reach_events = nwbfile.processing['behavior'].data_interfaces['ReachEvents']
    timestamps_re = reach_events.timestamps[:]  # the timing in s of each reach
    neural_data = nwbfile.acquisition['ElectricalSeries'].data
    sampling_rate = nwbfile.acquisition['ElectricalSeries'].rate

    # convert seconds to samples
    pre_samples = int(pre_time * sampling_rate)
    post_samples = int(post_time * sampling_rate)
    total_samples = neural_data.shape[0]

    if behavior is not None:
        clabels_orig = nwbfile.intervals['epochs'].to_dataframe()
        state_times = clabels_orig[clabels_orig['labels'].isin(behavior)][['start_time', 'stop_time']].to_numpy()
        # RAISE AN ERROR HERE ABOUT STATE_TIMES
        start = state_times[:, 0]
        stop = state_times[:, 1]
        in_any_interval = ((timestamps_re[:, None] >= start) & (timestamps_re[:, None] <= stop)).any(axis=1)
        timestamps_re = timestamps_re[in_any_interval]

    event_epochs = []

    for event_time in timestamps_re:
        # time to sample idx
        event_index = int(event_time * sampling_rate)
        start_idx = event_index - pre_samples
        end_idx = event_index + post_samples

        if start_idx >= 0 and end_idx < total_samples:
            epoch = neural_data[start_idx:end_idx, channel]
            event_epochs.append(epoch)

    event_epochs = np.array(event_epochs)  #(n_events, n_samples, n_channels)

    print('Number of intervals corresponding to the selected state(s):', state_times.shape[0])
    print('Number of wrist movement events that happened during the specified state(s):', len(event_epochs))

    ecog_data = {'signal': event_epochs, 'channel':channel, 'behavior': behavior, 'sampling_rate': sampling_rate, 'window': (pre_time, post_time)}

    return ecog_data

# can use the output of extract_neural_data as the input to avg_across_epochs
def avg_across_epochs(ecog_data):
    signal = ecog_data['signal']
    average_signal = np.nanmean(signal, axis=0) # shape = (n_samples, )
    ecog_data['signal'] = average_signal
    return ecog_data









