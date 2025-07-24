import numpy as np
from misc import demean
from SS.SS_utils import isint, plot_svc
from scipy.linalg import svd, cholesky

def tsdata_to_ssmo(X, pf, plotm=None):
    """
    Estimate optimal state-space model order using Bauer's Singular Value Criterion (SVC).

    Parameters
    ----------
    X : ndarray, shape (n, m, N)
        Observation time series, where
        n = number of variables,
        m = length of each trial,
        N = number of trials.
    pf : int or array_like of length 2
        Past/future horizons for canonical correlations.
        If scalar, p = f = pf.
        If length 2, pf = [p, f].
    plotm : None, str, optional
        Whether to plot the SVC curve.

    Returns
    -------
    mosvc : int
        Optimal model order according to Bauer's SVC.
    rmax : int
        Maximum possible model order given past/future horizons.

    Notes
    -----
    Bauer's SVC method estimates the SS model order by performing canonical
    correlations between past and future blocks of the time series and selecting
    the model order that minimizes the SVC criterion.

    The past/future horizons determine the block sizes and should satisfy
    p + f < m (length of each trial).
    """
    n, m, N = X.shape

    pf = np.atleast_1d(pf)
    assert isint(pf), "past/future horizon must be a 2-vector or scalar positive integer"
    if pf.size == 1:
        p = f = int(pf[0])
    elif pf.size == 2:
        p, f = pf
    else:
        raise ValueError("past/future horizon must be a 2-vector or scalar positive integer")

    assert p + f < m, "past/future horizon too large (or not enough data)"
    rmax = n * min(p, f)

    X = demean(X)
    mp = m - p
    mp1 = mp + 1
    mf = m - f
    mpf = mp1 - f

    M = N * mp
    M1 = N * mp1
    Mh = N * mpf

    # Build Xf
    Xf = np.zeros((n, f, mpf, N))
    for k in range(f):
        Xf[:, k, :, :] = X[:, (p + k):(mf + k + 1), :]
    Xf = Xf.reshape(n * f, Mh, order="F")

    # Build XP and Xp
    XP = np.zeros((n, p, mp1, N))
    for k in range(p):
        XP[:, k, :, :] = X[:, (p - k - 1):(m - k), :]
    Xp = XP[:, :, :mpf, :].reshape(n * p, Mh, order="F")
    XP = XP.reshape(n * p, M1, order="F")

    Wf = cholesky((Xf @ Xf.T) / Mh, lower=True)
    Wp = cholesky((Xp @ Xp.T) / Mh, lower=True)

    BETA = Xf @ np.linalg.pinv(Xp)
    assert np.all(np.isfinite(BETA)), "subspace regression failed"

    _, S, _ = svd(np.linalg.solve(Wf, BETA @ Wp))

    sval = S
    df = 2 * n * np.arange(1, rmax + 1)
    svc = -np.log(1 - np.append(sval[1:rmax], 0)) + df * (np.log(Mh) / Mh)

    morder = np.arange(rmax + 1)
    mosvc = morder[np.argmin(svc)]

    if plotm:
        import matplotlib.pyplot as plt
        plot_svc(sval, svc, mosvc, rmax, plotm)

    return mosvc, rmax

def tsdata_to_ss(X, pf, r=None, plotm=None):
    """
    Estimate an innovations-form state-space model from empirical time series data
    using Larimore's Canonical Correlations Analysis (CCA) subspace algorithm.

    Parameters
    ----------
    X : ndarray of shape (n, m, N)
        Observation time series with `n` variables, `m` time points, and `N` trials.
    pf : int or sequence of two ints
        Past/future horizons for canonical correlations. Can be:
        - A scalar: p = f = pf
        - A sequence (e.g., list/tuple/array) of two integers: [p, f]
    r : int or str, optional
        Model order:
        - If None or 'SVC', use Bauer's Singular Value Criterion to choose model order.
        - Otherwise, must be a positive integer <= n * min(p, f).
    plotm : bool or int, optional
        Whether to plot SVC curve (only used if r='SVC').
        - If True or an int: plot using matplotlib.
        - If False or None: no plot.

    Returns
    -------
    A : ndarray (r, r)
        State transition matrix.
    C : ndarray (n, r)
        Observation matrix.
    K : ndarray (r, n)
        Kalman gain matrix.
    V : ndarray (n, n)
        Innovations covariance matrix.
    Z : ndarray (r, m - p + 1, N)
        Estimated state process time series, shape (r, m - p + 1, N).
    E : ndarray (n, m - p, N)
        Estimated innovations process time series, shape (n, m - p, N).
    """
    n, m, N = X.shape

    pf = np.atleast_1d(pf)
    assert isint(pf), "past/future horizon must be a 2-vector or scalar positive integer"
    if pf.size == 1:
        p = f = int(pf[0])
    elif pf.size == 2:
        p, f = pf
    else:
        raise ValueError("past/future horizon must be a 2-vector or scalar positive integer")

    rmax = n * min(p, f)

    SVC = r is None or (isinstance(r, str) and r.lower() == 'svc')
    if not SVC:
        if not ((isinstance(r, (int, float)) or np.isscalar(r)) and isint([r]) and 0 < r <= rmax):
            raise ValueError(f"`r` must be an integer in [1, {rmax}]")

    X = demean(X)

    mp  = m - p
    mp1 = mp + 1
    mf  = m - f
    mpf = mp1 - f

    M  = N * mp
    M1 = N * mp1
    Mh = N * mpf

    # Build Xf
    Xf = np.zeros((n, f, mpf, N))
    for k in range(f):
        Xf[:, k, :, :] = X[:, (p + k):(mf + k + 1), :]
    Xf = Xf.reshape(n * f, Mh, order="F")

    # Build XP and Xp
    XP = np.zeros((n, p, mp1, N))
    for k in range(p):
        XP[:, k, :, :] = X[:, (p - k - 1):(m - k), :]
    Xp = XP[:, :, :mpf, :].reshape(n * p, Mh, order="F")
    XP = XP.reshape(n * p, M1, order="F")

    # Cholesky of forward and backward covariance
    Wf = np.linalg.cholesky((Xf @ Xf.T) / Mh)
    Wp = np.linalg.cholesky((Xp @ Xp.T) / Mh)

    # Regression and SVD
    BETA = Xf @ np.linalg.pinv(Xp)
    _, S, Vt = np.linalg.svd(np.linalg.solve(Wf, BETA @ Wp))
    sval = S

    if SVC:
        df = 2 * n * np.arange(1, rmax + 1)
        svc = -np.log(1 - np.append(sval[1:rmax], 0)) + df * (np.log(Mh) / Mh)
        morders = np.arange(0, rmax + 1)
        r = morders[np.argmin(svc)]
        if r == 0:
            raise ValueError("SVC model order is zero")
        if r == rmax:
            print("*** WARNING: SVC model order is maximum ('pf' may have been set too low)")
        if plotm:
            plot_svc(sval, svc, r, rmax, plotm)

    # Kalman state sequence Z
    Z = (np.diag(np.sqrt(sval[:r])) @ Vt[:r, :] @ np.linalg.inv(Wp)) @ XP
    Z = Z.reshape(r, mp1, N, order="F")

    # Estimate observation matrix C
    Z_reshape = Z[:, :mp, :].reshape(r, -1, order="F")
    X_reshape = X[:, p:m, :].reshape(n, -1, order="F")
    C = X_reshape @ np.linalg.pinv(Z_reshape)

    # Innovations and covariance
    E = X_reshape - C @ Z_reshape
    V = (E @ E.T) / (M - 1)

    # Estimate A and K
    Z2 = Z[:, 1:mp1, :].reshape(r, -1, order="F")
    regmat = np.vstack((Z_reshape, E))
    AK = Z2 @ np.linalg.pinv(regmat)
    A = AK[:, :r]
    K = AK[:, r:]

    E = E.reshape(n, mp, N, order="F")

    return A, C, K, V, Z, E

# Example usage
if __name__ == "__main__":
    
    np.random.seed(0)

    # Parameters
    n = 4       # Number of channels
    m = 1000    # Number of time points
    N = 10      # Number of trials

    # Time vector
    t = np.arange(1, m + 1)

    # Preallocate
    X = np.zeros((n, m, N))

    # Generate data
    for trial in range(N):
        for ch in range(n):
            freq = 0.01 * (1 + (ch + 1)) 
            phase = np.pi / 8 * (trial + 1)
            amp = 1 + 0.5 * (ch + 1)
            signal = amp * np.sin(2 * np.pi * freq * t + phase)
            noise = 0.5 * np.random.randn(m)
            X[ch, :, trial] = signal + noise

    # Past and future horizons
    pf = 5  
    # Model order
    r_optimal, rmax = tsdata_to_ssmo(X, pf, plotm=True)
    print(f"Optimal model order: {r_optimal}, Maximum possible order: {rmax}")

    A, C, K, V, Z, E = tsdata_to_ss(X, pf, r=r_optimal, plotm=True)
    # print("State transition matrix A:\n", A)
    # print("Observation matrix C:\n", C)
    # print("Kalman gain matrix K:\n", K)
    # print("Innovations covariance matrix V:\n", V)

    # Or, equivalently:
    A, C, K, V, Z, E = tsdata_to_ss(X, pf, r="SVC", plotm=True)
    print(f"Optimal model order: {A.shape[0]}.")