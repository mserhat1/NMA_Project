from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

# pcs =
# top_joints = n gives the n joints that contribute to the first PC the most
# top_kps = n gives the n keypoints that contribute to the first PC the most
# returns the array containing principal components in order of explained variance
def keypoints_pca(spatial_data, interp_reference=None, plot=True):
    keypoints = spatial_data.keys()

    kp_data = []
    kp_names = []

    for kp in keypoints:
        kp_data.append(spatial_data[kp]['data'])
        kp_names += [f'{kp}_X', f'{kp}_Y']

    kp_data = np.hstack(kp_data)
    
    # interpolate
    if interp_reference is not None:
        kp_interp = time_interpolate(kp_interp, interp_reference)

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

def project_onto_pcs(data, n_pcs):
  ts_data = data['kp_data']
  pcs = data['components']
  top_pcs = pcs[:n_pcs, :]

  projected = ts_data @ top_pcs.T

  return projected # ndarray (n_samples, n_pcs)

def time_interpolate(spatial, neural, kind='linear'):
    n_original = spatial.shape[0]
    n_target = neural.shape[0]

    x_original = np.linspace(0, 1, n_original)
    x_new = np.linspace(0, 1, n_target)

    interpolated_data = np.empty((n_target, spatial.shape[1]))

    for i in range(spatial.shape[1]):
        f = interp1d(x_original, spatial[:, i], kind=kind)  # or 'cubic'
        interpolated_data[:, i] = f(x_new)

    return interpolated_data

    






