import numpy as np
def RRR(K, N, r):
    # K: (T, d) demeaned PCs
    # N: (T, D) demeaned neural targets

    # full least‑squares map
    B_full = np.linalg.pinv(K) @ N  # (d, D)

    # truncated SVD of the fitted predictions
    U, S, Vt = np.linalg.svd(K @ B_full, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], np.diag(S[:r]), Vt[:r, :]

    # reduced‑rank factors
    A = np.linalg.pinv(K) @ Ur @ Sr  # (d, r)
    C = Vr.T  # (D, r)

    # inference: z =  K  @ A ;   N_est = z @ C.T

    return A, C

def RRR_modified(K, N, r=None, *, scale_X=False, scale_Y=False, ridge=None,
        gain=False, return_misc=False):
    """
    Reduced-rank regression with optional variance scaling, ridge penalty
    and post-fit diagonal gain.

    K : (T,d)   predictors (demeaned)
    N : (T,D)   responses  (demeaned)
    r : int     desired rank (default: min(d,D))
    """

    T, d = K.shape
    _, D = N.shape
    assert r <= K.shape[1]  # desired rank  (≤ d)

    # -- optional standardisation --
    K0, N0 = K.copy(), N.copy()
    Km, Nm = K.mean(0), N.mean(0)
    if scale_X:
        Ks = K.std(0) + 1e-12
        K0 = (K0 - Km) / Ks
    else:
        K0 -= Km
    if scale_Y:
        Ns = N.std(0) + 1e-12
        N0 = (N0 - Nm) / Ns
    else:
        N0 -= Nm

    # -- full (or ridged) least-squares map --
    if ridge is None or ridge == 0:
        B_full = np.linalg.pinv(K0) @ N0
    else:
        Bt = np.linalg.inv(K0.T @ K0 + ridge*np.eye(d)) @ K0.T @ N0
        B_full = Bt

    # -- truncated SVD of fitted values --
    U,S,Vt = np.linalg.svd(K0 @ B_full, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], np.diag(S[:r]), Vt[:r, :]

    # -- core RRR factors --
    A = np.linalg.pinv(K0) @ Ur @ Sr
    C = Vr.T

    # -- optional post-fit gain --
    if gain:
        z = K0 @ A
        G = np.diag(np.linalg.lstsq(z, N0, rcond=None)[0])  # r×r
        C = C @ G

    misc = dict(K_mean=Km, N_mean=Nm)
    if scale_X: misc['K_std'] = Ks
    if scale_Y: misc['N_std'] = Ns
        return A, C, Ns, Nm
    if return_misc:
        return A, C, misc
    return A, C
