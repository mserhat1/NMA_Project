from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# pcs =
# top_joints = n gives the n joints that contribute to the first PC the most
# top_kps = n gives the n keypoints that contribute to the first PC the most
def keypoints_pca(spatial_data, time_window='all', pcs='all', top_kps=4):
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

    if pcs == 'all':
        pcs = len(kp_names)

    pca = PCA(n_components=pcs)
    pca.fit_transform(kp_scaled)

    loadings = np.abs(pca.components_[0])
    top_indices = np.argsort(loadings)[::-1]
    top_features = [kp_names[i] for i in top_indices[:top_kps]]
    print("Top movement contributors in the first PC:", top_features)

    pc_vector = np.arange(1, pcs + 1)
    variance_per_pc = pca.explained_variance_ratio_
    plt.figure(figsize=(12, 6))
    plt.plot(pc_vector, variance_per_pc, marker='o', fillstyle='full')
    plt.xlabel("Principal components")
    plt.ylabel("Explained variance ratio")
    plt.xticks(pc_vector)
    plt.title("Variance ratio explained by each principal component")





