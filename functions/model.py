import numpy as np
def RRR(K, N):
    # K: (T, d) demeaned PCs
    # N: (T, D) demeaned neural targets
    r = K.shape[1]  # desired rank  (≤ d)

    # full least‑squares map
    B_full = np.linalg.pinv(K) @ N  # (d, D)

    # truncated SVD of the fitted predictions
    U, S, Vt = np.linalg.svd(K @ B_full, full_matrices=False)
    Ur, Sr, Vr = U[:, :r], np.diag(S[:r]), Vt[:r, :]

    # reduced‑rank factors
    A = np.linalg.pinv(K) @ Ur @ Sr  # (d, r)
    C = Vr.T  # (D, r)

    # inference: z =  X  @ A ;   Ŷ = z @ C.T

    return A, C