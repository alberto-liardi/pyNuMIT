import numpy as np

from VAR.VAR_utils import *
from misc import demean, isbad
from numpy.linalg import LinAlgError
from scipy.stats import f

from warnings import warn


def info_criteria(L, k, m):
    """
    Calculate Akaike (AIC), Bayesian (BIC), and Hannan-Quinn (HQC) information criteria.
    
    Parameters:
    L : float or array-like
        Log-likelihood of the model.
    k : float or array-like
        Number of free parameters in the model.
    m : float or array-like
        Number of observations.

    Returns:
    tuple
        aic : float or array-like
            Akaike Information Criterion (AIC).
        bic : float or array-like
            Bayesian Information Criterion (BIC).
        hqc : float or array-like
            Hannan-Quinn Criterion (HQC).
    """
    K = k / m  # Adjusted number of parameters per observation
    
    # Calculate the information criteria
    aic = -2 * L + 2 * K                 # Akaike's AIC
    bic = -2 * L + K * np.log(m)         # Schwarz' BIC
    hqc = -2 * L + 2 * K * np.log(np.log(m))  # Hannan-Quinn IC
    
    return aic, bic, hqc


def tsdata_to_varll(X, p, regmode="OLS", verb=False):
    """
    Calculate log-likelihood, and related metrics for VAR models.
    
    Parameters:
    X : ndarray
        Time series data of shape (n, m, N), where:
        n = number of variables,
        m = number of observations,
        N = number of trials.
    p : int
        Maximum model order.
    regmode : str
        Regression mode ('OLS' supported).
    verb : bool, optional
        Verbose output (default: False).
    
    Returns:
    tuple
        LL : ndarray
            Log-likelihood values.
        Lk : ndarray
            Number of free parameters.
        Lm : ndarray
            Effective sample size.
    """
    if len(X.shape) == 2:
        X = X[:,:,np.newaxis]
    n, m, N = X.shape

    assert isinstance(p, int) and p > 0 and p < m, \
        "Maximum model order must be a positive integer less than the number of observations."

    # Demean data (removing the mean)
    X = demean(X)

    p1 = p + 1
    pn = p * n
    p1n = p1 * n

    I = np.eye(n)

    # Store lags
    XX = np.zeros((n, p1, m + p, N))
    for k in range(p + 1):
        XX[:, k, k:k + m, :] = X

    # Initialize outputs
    LL = np.full(p1, np.nan)
    Lk = np.full(p1, np.nan)
    Lm = np.full(p1, np.nan)

    # Order zero log-likelihood
    M = N * m
    E = X.reshape(n, M, order="F")  # Residuals are just the series itself
    # V = np.cov(E)  # Covariance matrix (ML estimator)
    V = (E@E.T)/M

    LL[0] = -np.linalg.slogdet(V)[1] / 2
    Lk[0] = 0
    Lm[0] = M

    # Loop through model orders
    if regmode == 'OLS':
        for k in range(1, p+1):
            if verb:
                print(f"model order = {k}")

            k1 = k + 1
            M = N * (m - k)

            X0 = XX[:, 0, k:m, :].reshape(n, M, order="F")
            XL = XX[:, 1:k1, k:m, :].reshape(n * k, M, order="F")

            try:
                A = np.linalg.lstsq(XL.T, X0.T, rcond=None)[0].T
            except LinAlgError as e:
                if verb:
                    print(f"WARNING: VAR estimation failed for model order {k}: {e}")
                continue

            if isbad(A):
                if verb:
                    print(f"WARNING: VAR estimation failed for model order {k}: bad coefficients")
                continue

            E = X0 - np.dot(A, XL)  # Residuals
            # V = np.cov(E)  # Residual covariance matrix (ML estimator)
            V = (E@E.T)/M

            LL[k1-1] = -np.linalg.slogdet(V)[1] / 2
            Lk[k1-1] = k * n * n
            Lm[k1-1] = M
    else:
        raise NotImplementedError

    return LL, Lk, Lm


def tsdata_to_varmo(X, p, regmode="OLS", alpha=[0.01, 0.05], verb=False):
    """
    Determine optimal VAR model order using information criteria and likelihood ratio tests.

    Parameters:
        X (numpy.ndarray): The input data array (n x m x N).
        p (int): Maximum model order.
        regmode (str): Regression mode ('OLS').
        alpha (float or list, optional): Significance levels for likelihood ratio test. Defaults to [0.01, 0.05].
        verb (int, optional): Verbosity level. Defaults to 0.

    Returns:
        moaic (int): Optimal model order according to Akaike Information Criterion (AIC).
        mobic (int): Optimal model order according to Bayesian Information Criterion (BIC).
        mohqc (int): Optimal model order according to Hannan-Quinn Criterion (HQC).
        molrt (int): Optimal model order according to likelihood ratio test (LRT).
    """
    if np.isscalar(alpha):
        alpha = [alpha, alpha]

    n = X.shape[0]

    # Get log-likelihoods, PACFs at VAR model orders 1 to p
    LL, Lk, Lm = tsdata_to_varll(X, p, regmode, verb)

    # Calculate information criteria
    aic, bic, hqc = info_criteria(LL, Lk, Lm)

    # Calculate optimal model orders according to information criteria
    morder = np.arange(p + 1)
    moaic = morder[np.nanargmin(aic)]
    mobic = morder[np.nanargmin(bic)]
    mohqc = morder[np.nanargmin(hqc)]

    # Sequential likelihood ratio F-test for optimal model order
    lambda_ = 2 * np.diff(Lm * LL)  # LR test statistics
    df = n * n  # Degrees of freedom
    df2 = Lm[1:] - n * np.arange(1, p + 1) - 1
    lrpval = 1 - f.cdf(lambda_ / df, df, df2)
    lrpval[df2 <= 0] = np.inf  # Give up if F-test fails
    lralpha = alpha[0] / p  # Bonferroni adjustment to significance level

    molrt = 0
    for k in range(p, 0, -1):
        if lrpval[k - 1] < lralpha:
            molrt = k
            break

    return moaic, mobic, mohqc, molrt


def fit_var1(data, mode="OLS", detrend=True, lag=1):
    """
    Fits a VAR(1) model: X_{t+1} = A*X_t + epsilon_t
    Parameters:
        data (numpy.ndarray): 
            Multivariate time series, shape (n, T, m), where n is the number of channels, T is the number of time steps, m the number of trials.
        mode (str):
            Estimation mode: "OLS" (default).
        mode (str):
            Estimation mode: "OLS" (default).
        detrend (bool):
            Whether to standardise the input data to mean zero and unit variance. Defaults to True.
    Returns:
        A (numpy.ndarray): 
            Transition matrix of shape (n, n).
        V (numpy.ndarray): 
            Covariance matrix of residuals, shape (n, n).
    """

    assert data.shape[0]<data.shape[1], f"Not enough timepoints ({data.shape[1]}) for {data.shape[0]} channels. Did you mean to transpose the data?"
    if len(data.shape) == 2:
        data = data[:,:,np.newaxis]
    
    n, T, m = data.shape
    M = (T-lag)*m

    if detrend: data = demean(data)

    if mode == "OLS":
        # Create lagged data for each trial, then reshape
        X_t  = data[:,:-lag,:].reshape(n, M, order="F")  # All timepoints except the last (past)
        X_tp1 = data[:,lag:,:].reshape(n, M, order="F")  # All timepoints except the first (future)
        
        try:
            np.linalg.cholesky(X_t @ X_t.T)
        except np.linalg.LinAlgError:
            raise AssertionError("Covariance matrix of the data is singular!")

        # Estimate A using least squares: A = (X_tp1 * X_t.T) * (X_t * X_t.T)^-1
        A = (X_tp1 @ X_t.T) @ np.linalg.inv(X_t @ X_t.T)

        # Calculate residuals
        E = X_tp1 - A @ X_t 

        # Compute residual covariance
        # V = np.cov(E.T, rowvar=False)
        V = (E@E.T)/(M-1)
        
    else:
        raise NotImplementedError

    return A, V, E


def fit_var(data, p=None, maxp=None, mode='OLS', detrend=True):
    """
    Fits a VAR(p) model: X_{t+1} = A_1*X_t + A_2*X_{t-1} + ... + A_p*X_{t-p} + epsilon_t
    Parameters:
        data (numpy.ndarray): 
            Multivariate time series, shape (n, T, m), where n is the number of channels, T is the number of time steps, m the number of trials.
        p (int): 
            Model order of the VAR. If None, the order is estimated using HQC criterium (see ts_to_varmo).
        maxp (int): 
            Maximum model order of the VAR in the HQC estimation. Only used if p==None.
        mode (str):
            Estimation mode: "OLS" (default).
        detrend (bool):
            Whether to standardise the input data to mean zero and unit variance. Defaults to True.
    Returns:
        A (numpy.ndarray): 
            Transition matrices of shape (n, n, p).
        V (numpy.ndarray): 
            Covariance matrix of residuals, shape (n, n).
    """

    assert data.shape[0]<data.shape[1], f"Not enough timepoints ({data.shape[1]}) for {data.shape[0]} channels. Did you mean to transpose the data?"
    # n, T, m = data.shape if len(data.shape) == 3 else (*data.shape, 1)
    if len(data.shape) == 2:
        data = data[:,:,np.newaxis]
    n, T, m = data.shape
    
    if detrend: data = demean(data)

    # if model order is not specified, find the best one
    if p is None:
        if maxp is None:
            maxp = int(12 * (T/100.)**(1./4))
        else: 
            assert isinstance(maxp, int) and maxp > 0, "Maximum model order must be a positive integer."
        p = tsdata_to_varmo(data, maxp, mode)[3] # using HQC criterium by default
        if p<1:
            warn(f"VAR model order selection has failed. Setting it to 1.")
            p=1
    elif p==1: 
        # call VAR1 fitness (faster version)
        return fit_var1(data, mode)
    else:
        assert isinstance(p, int) and p > 0, "Model order must be a positive integer."

    M = m*(T-p)

    if mode == 'OLS':
        
        # the usable observations (i.e., lose the first p)
        obs = np.arange(p, T)

        # Reshape data for unlagged observations
        X0 = data[:, obs].reshape(n, M, order='F')

        # XL for lagged observations
        XL = np.zeros((n, p, M))
        for k in range(p):
            XL[:, k, :] = data[:, obs-k-1].reshape(n, M, order='F')

        # Stack lagged observations
        XL = XL.reshape(p * n, M, order='F')
        
        # Ordinary Least Squares (via QR decomposition)
        A = np.linalg.lstsq(XL.T, X0.T, rcond=None)[0].T


        # Calculate the residuals   
        E = X0 - A @ XL  # residuals

        # Calculate residual covariance
        V = (E@E.T)/(M-1)

        # Reshape A so that A[:,:,k] is the k-lag coefficients matrix
        A = A.reshape(n, n, p, order='F')
        E = np.concatenate((np.full((n, p, m), np.nan), E.reshape(n, T - p, m, order="F")), axis=1)

    else:   
        raise NotImplementedError

    return A, V, p, E