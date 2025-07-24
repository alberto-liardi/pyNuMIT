import numpy as np
from warnings import warn

from numpy.linalg import slogdet

from VAR.VAR_fitness import tsdata_to_varmo
from SS.SS_utils import ss_info, ss_to_cpsd, bandlimit
from SS.SS_fitness import tsdata_to_ss


def StateSpaceEntropyRate(X, Fs, downsampling="yes", band=None, varmomax=20):
    """
    Estimate the state-space entropy rate (CSER) of a time series.

    Parameters
    ----------
    X : ndarray
        Input time series data (2D or 3D).
    Fs : float
        Sampling frequency of the time series.
    downsampling : str, optional
        Downsampling method (default is "yes").
    band : ndarray, optional
        Frequency bands for bandlimited CSER estimation (default is None).
    varmomax : int, optional
        Maximum VAR model order (default is 20).

    Returns
    -------
    CSER : ndarray(D,) if band is not provided, otherwise ndarray(D, band.shape[0]+1)
        Estimated CSER values for each channel (and frequency band).
        If bands are provided, each row corresponds to a channel and the first column corresponds
        to the broadband CSER, while each subsequent column corresponds to a specific frequency band.
    """

    if X.ndim not in [2, 3]:
        raise ValueError("Input must be a 2 or 3D data matrix")

    if X.ndim == 2:
        X = X[:, :, np.newaxis]

    D, T, M = X.shape
    H = 0

    if T < D:
        warn("Warning: More spatial than temporal dimensions. Need to transpose?")

    if downsampling == "yes" and Fs > 200:
        k = int(np.floor(Fs / 200))
        Fs = Fs / k
        X = X[:, ::k, :]

    CSER = (
        np.ones((D, 1)) * np.nan
        if band is None
        else np.ones((D, band.shape[0] + 1)) * np.nan
    )

    def H_fun(C):
        return 0.5 * slogdet(2 * np.pi * np.exp(1) * C)[1]

    for d in range(D):
        failed = False
        y = X[d : d + 1, :, :]

        varmoaic, varmobic, varmohqc, varmolrt = tsdata_to_varmo(y, varmomax)

        if varmohqc < 1:
            pf = 1
            ssmo = 2
        else:
            pf = 2 * varmohqc
            ssmo = "SVC"

        try:
            A, C, K, V, _, _ = tsdata_to_ss(y, pf, r=ssmo, plotm=False)
            assert V.shape == (1, 1)
            info = ss_info(A, C, K, V, 0)
        except Exception as e:
            print(f"State space model fitting failed for channel {d}: {e}")
            H = np.nan
            failed = True
        else:
            failed = info["error"] != 0
            if failed:
                H = np.nan

        if not failed:
            CSER[d, 0] = H_fun(V)

        if band is not None:
            if not failed:
                fres = 1000
                S = ss_to_cpsd(A, C, K, V, fres)[0]
                H_freq = np.array([H_fun(S[:, :, i]) for i in range(fres + 1)])

                for j in range(band.shape[0]):
                    band_j = band[j].copy()
                    if np.isinf(band_j[1]):
                        band_j[1] = int(np.floor(Fs / 2.0))
                    CSER[d, j + 1] = bandlimit(H_freq, 0, Fs, band_j)

    return CSER if band is not None else CSER.reshape(-1)


if __name__ == "__main__":

    # generate some synthetic data:
    n, m, N = 4, 1000, 10  # channels, time points, trials
    t = np.arange(m)  # time vector
    freqs = 0.01 * (1 + np.arange(1, n + 1))[:, None, None]
    phases = (np.pi / 8) * np.arange(1, N + 1)[None, None, :]
    amps = (1 + 0.5 * np.arange(1, n + 1))[:, None, None]
    T = t[None, :, None]
    signal = amps * np.sin(2 * np.pi * freqs * T + phases)
    noise = 0.5 * np.random.randn(n, m, N)
    X = signal + noise

    # test CSER without bands
    cser = StateSpaceEntropyRate(X, Fs=200)
    print("CSER:", cser)

    # test CSER with bands
    cser_bands = StateSpaceEntropyRate(
        X,
        Fs=200,
        downsampling="yes",
        band=np.array([[1, 4], [4, 8], [8, 12], [12, 25]]),
    )
    print("CSER with bands:", cser_bands)
