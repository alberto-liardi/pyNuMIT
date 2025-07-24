import numpy as np
from numpy.linalg import slogdet, LinAlgError, inv
from scipy.signal import butter, filtfilt
import pickle

def arrayfy(data_dict):
    """
    Convert an array of dictionaries to a matrix of their values.

    Parameters:
        data_dicts (dict or list of dict): A dictionary or an array (or list) of dictionaries with the same keys.
        
    Returns:
        np.ndarray: A matrix where each column represents the values from one dictionary.
    """
    if isinstance(data_dict, dict): data_dict = [data_dict]
    return np.array([np.fromiter(d.values(), dtype=float) for d in data_dict]).T
    # return np.array([[d[k] for d in data_dict] for k in data_dict[0].keys()], dtype=float)


def dictify(data, labels=None):
    """
    Convert a matrix to a dictionary with specified labels.

    Parameters:
        data (np.ndarray): A matrix where each column represents a variable.
        labels (list of str, optional): A list of labels for each column. If None, default labels are used.

    Returns:
        dict: A dictionary where keys are labels and values are the corresponding columns from the data.
    """    
    if not labels:
        labels = ["rtr", "rta", "rtb", "rts", "xtr", "xta", "xtb", "xts",
                    "ytr", "yta", "ytb", "yts", "str", "sta", "stb", "sts"]
    else:
        assert len(data) == len(labels), "Length of data and labels must match."
    
    return {label: data[i] for i, label in enumerate(labels)}


def CompQuantile(distr, value):
    """
    Compute the quantiles of given values for input distributions.

    Parameters:
        distr (ndarray): [M, N] array where each row represents a distribution with N samples.
        value (ndarray): [M, L] array where each row contains L values for which quantiles are computed.

    Returns:
        quantile (ndarray): [M, L] array where each element represents the quantile of the corresponding
                            value in 'value' relative to the corresponding distribution in 'distr'.
    """
    if len(distr.shape)==len(value.shape)==1:
        distr = distr[np.newaxis, :]
        value = value[np.newaxis, :] 
    if distr.shape[0] != value.shape[0]:
        raise ValueError("Dimensions not consistent!")

    quantile = np.zeros_like(value, dtype=float)

    for l in range(value.shape[1]):
        nless = np.nansum(distr < value[:, l][:, None] - 1e-8, axis=1)
        nequal = np.nansum(np.abs(distr - value[:, l][:, None]) < 1e-8, axis=1)
        quantile[:, l] = (nless + 0.5 * nequal) / distr.shape[1]

    return quantile


def Sigmoid(x, M):
    """
    Sigmoid transformation.

    Parameters:
        x (float): Input value.
        M (float): Scaling factor.

    Returns:
        float: Transformed value.
    """
    return M / (1 + np.exp(-x))


def h(S):
    """
    Computes the differential entropy of a multivariate Gaussian.
    
    Parameters:
    -----------
    S : ndarray
        Covariance matrix of the system (assumed to be positive definite).
    
    Returns:
    --------
    float
        Half the log-determinant of the covariance matrix.
    """
    sign, logdet = slogdet(S)
    if sign <= 0:
        raise LinAlgError("Matrix is not positive definite.")
    return 0.5 * logdet


def conditional_cov(Sigma, idx_A, idx_B):
    """
    Compute conditional covariance S_{A|B} = S_AA - S_AB S_BB^{-1} S_BA
    
    Parameters:
    -----------
    Sigma : ndarray
        Covariance matrix of the system (assumed to be positive definite).  
    idx_A : list
        Indices of the first variable.
    idx_B : list
        Indices of the second variable.
    
    Returns:
    --------
    ndarray
        Conditional covariance matrix S_{A|B}.
    """
    Sigma_AA = Sigma[np.ix_(idx_A, idx_A)]
    Sigma_AB = Sigma[np.ix_(idx_A, idx_B)]
    Sigma_BB = Sigma[np.ix_(idx_B, idx_B)]
    Sigma_BA = Sigma_AB.T
    return Sigma_AA - Sigma_AB @ inv(Sigma_BB) @ Sigma_BA


def RedMMI(I1, I2, *args):
    """
    Computes redundancy between two variables using the Minimal Mutual Information (MMI).
    """
    return min(I1, I2)


def doubleRedMMI(I1, I2, I3, I4, *args):
    """
    Computes double redundancy between two variables using the Minimal Mutual Information (MMI).
    """
    return min(I1, I2, I3, I4)


def RedCCS(I1, I2, I12):
    """
    Computes redundancy between two variables using the Common Change in Surprisal (CCS).
    """
    c = I12 - I1 - I2
    if np.all(np.sign([I1, I2, I12, -c]) == np.sign(I1)):
        return -c
    return 0


def doubleRedCCS(I1, I2, I3, I4, Ixtab, Iytab, Ixyta, Ixytb, 
                 Ixytab, Rxyta, Rxytb, Rabtx, Rabty, Rxytab, Rabtxy):
    """
    Computes double redundancy between two variables using the Common Change in Surprisal (CCS).
    """
    coinfo = (-I1 - I2 - I3 - I4 +
              Ixtab + Iytab + Ixyta + Ixytb - Ixytab +
              Rxyta + Rxytb - Rxytab +
              Rabtx + Rabty - Rabtxy)

    signs = np.sign([I1, I2, I3, I4, coinfo])
    if np.all(signs == signs[0]):
        return coinfo
    return 0


def get_gaussian_cov(data, lag=1, detrend=True):
    """
    Get the vectors of the time series data for arbitrary number of channels

    Parameters
    ----------
    data : array
        Array of shape (n_channels, timepoints)
    lag : int
        Time lag

    Returns
    -------
    past : array
        Array of shape (n_channels, T - lag) with past values
    future : array
        Array of shape (n_channels, T - lag) with future values
    """
    if detrend: data = demean(data) 
    past = data[:, :-lag] 
    future = data[:, lag:]
    if len(data.shape) == 3:
        past = past.reshape(past.shape[0], past.shape[1]*past.shape[2], order='F')
        future = future.reshape(future.shape[0], future.shape[1]*future.shape[2], order='F')
    cov = np.cov(np.vstack([past, future]))
    return cov


def demean(data):
    """
    Demean and standardise time series to mean 0 and unit variance.

    Parameters:
        data (numpy.ndarray): 
            Multivariate time series, shape (n, T, m), where n is the number of channels, 
            T is the number of time steps, m the number of trials.
    Returns:
        (numpy.ndarray): 
            Standardised multivariate time series of shape (n, T, m).
    """
    n, T, m = data.shape if len(data.shape) == 3 else (*data.shape, 1)
    data2d = data.reshape(n,T*m, order="F")

    mean = np.mean(data2d, axis=1, keepdims=True)
    std = np.std(data2d, axis=1, keepdims=True, ddof=1)

    data_dem = ((data2d - mean) / std)
    return data_dem.reshape(n, T, m, order="F") if len(data.shape) == 3 else data_dem.reshape(n, T, order="F")


def isbad(x):
    """
    Determine whether an array is "bad".

    Parameters:
    x : array-like
        Input array to check.

    Returns:
    bool
        True if the array is "bad", False otherwise.
    """
    x = np.asarray(x)  # Ensure input is a NumPy array

    return not np.all(np.isfinite(x))  # At least one NaN or Inf


def saveData(data, file_path):
    """
    Save a variable to a file with pickle.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def loadData(file_path):
    """
    Open and load a pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

def low_pass_filter(data, down_s, fs):
    """
    Applies a low-pass filter to downsampled data.

    Parameters:
    -----------
    data : ndarray
        Multivariate time series data with shape (n_channels, n_timepoints, n_trials).
    down_s : int
        Downsampling factor.
    fs : int
        Sampling frequency of the original data.

    Returns:
    --------
    filtered_data : ndarray
        Filtered and downsampled data with reduced dimensions.
    """
    nyquist = fs / 2
    cutoff = nyquist / down_s
    b, a = butter(4, cutoff / nyquist, btype='low')
    return filtfilt(b, a, data, axis=1)[:,::down_s,:]