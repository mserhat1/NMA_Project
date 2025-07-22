from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# pcs =
# top_joints = n gives the n joints that contribute to the first PC the most
# top_kps = n gives the n keypoints that contribute to the first PC the most
# returns the array containing principal components in order of explained variance
def keypoints_pca(spatial_data, time_window='all', plot=True):
    keypoints = spatial_data.keys()

    kp_data = []
    kp_names = []

    for kp in keypoints:
        if time_window == 'all':
            kp_data.append(spatial_data[kp]['data'])
        else:
            rate = spatial_data[kp]['sampling_rate']
            t1, t2 = time_window
            assert t1 >= 0 and t2 <= spatial_data[kp]['data'].shape[
                0] / rate and t1 < t2, 'We need 0 <= t1 < t2 <= duration of data'
            s1 = int(t1 * rate)
            s2 = int(t2 * rate)
            kp_data.append(spatial_data[kp]['data'][s1:s2, :])

        kp_names += [f'{kp}_X', f'{kp}_Y']

    kp_data = np.hstack(kp_data)

    # interpolate over nan's
    kp_interp = kp_data.copy()
    for i in range(kp_data.shape[1]):
        nans = np.isnan(kp_interp[:, i])
        indices = np.arange(kp_interp.shape[0])
        kp_interp[:, i] = np.interp(indices, indices[~nans], kp_data[:, i][~nans])

    # scaling
    kp_scaled = StandardScaler().fit_transform(kp_interp)

    pca = PCA(n_components=len(kp_names))
    pca.fit_transform(kp_scaled)

    if plot:
        pc_vector = np.arange(1, len(kp_names) + 1)
        variance_per_pc = pca.explained_variance_ratio_
        plt.figure(figsize=(12, 6))
        plt.plot(pc_vector, variance_per_pc, marker='o', fillstyle='full')
        plt.xlabel("Principal components")
        plt.ylabel("Explained variance ratio")
        plt.xticks(pc_vector)
        plt.title("Variance ratio explained by each principal component")

    pca_output = {'components': pca.components_, 'kp_data': kp_scaled, 'pca': pca}

    return pca_output

# n_components can be an integer (this specifies the exact no. of components returned)
# it can be a 0 < float < 1, which specifies the percentage of variance you want your components to explain.
def ecog_pca(ecog_data, n_components, time_window='all',  plot=True):
    signal = ecog_data['signal'] # (n_samples, n_channels)
    rate = ecog_data['sampling_rate']

    if time_window != 'all':
        t1, t2 = time_window
        s1 = int(t1 * rate)
        s2 = int(t2 * rate)
        signal = signal[s1:s2, :]

    # interpolate over nan's
    signal_interp = signal.copy()
    for i in range(signal_interp.shape[1]):
        nans = np.isnan(signal_interp[:, i])
        indices = np.arange(signal_interp.shape[0])
        signal_interp[:, i] = np.interp(indices, indices[~nans], signal[:, i][~nans])

    # scaling
    signal_scaled = StandardScaler().fit_transform(signal_interp)

    pca = PCA(n_components=n_components)
    pca.fit_transform(signal_scaled)

    n_pcs = pca.components_.shape[0]

    if plot:
        pc_vector = np.arange(1, n_pcs + 1)
        variance_per_pc = pca.explained_variance_ratio_
        plt.figure(figsize=(12, 6))
        plt.plot(pc_vector, variance_per_pc, marker='o', fillstyle='full')
        plt.xlabel("Principal components")
        plt.ylabel("Explained variance ratio")
        plt.xticks(pc_vector)
        plt.title("Variance ratio explained by each principal component")

    pca_output = {'components': pca.components_, 'kp_data': signal_scaled, 'pca': pca}

    return pca_output






