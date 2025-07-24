import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import warnings
from misc import demean


def var_decay(A, dfac):
    """
    Exponentially decay VAR coefficients matrix A by a factor dfac.
    
    Parameters:
        A (numpy.ndarray): VAR coefficients matrix (can be 1D or 3D).
        dfac (float): Decay factor.
        
    Returns:
        numpy.ndarray: Decayed VAR coefficients matrix.
    """
    if A.ndim == 1:  # A is a vector
        p = len(A)
        f = dfac
        for k in range(p):
            A[k] = f * A[k]
            f = dfac * f
    else:  # A is a 3D matrix
        if len(A.shape)==2: A = A[:,:,np.newaxis]
        p = A.shape[2]
        f = dfac
        for k in range(p):
            A[:, :, k] = f * A[:, :, k]
            f = dfac * f
    return A



def specnorm(A, newrho=None):
    """
    Calculate or adjust the spectral norm of VAR or VMA coefficients.

    Parameters:
        A (numpy.ndarray): VAR (or -VMA) coefficients array.
        newrho (float, optional): New spectral norm value to adjust coefficients to.

    Returns:
        If `newrho` is not provided:
            rho (float): Spectral norm of the input coefficients.
        If `newrho` is provided:
            A (numpy.ndarray): Adjusted coefficients with the specified spectral norm.
            rho (float): Original spectral norm of the input coefficients.
    """
    if A.ndim == 1: 
        p = len(A)
        n = 1
    else:  # A is a 3D array
        if len(A.shape)==2: A = A[:,:,np.newaxis]
        n, n1, p = A.shape
        assert n1 == n, "VAR/VMA coefficients matrix has bad shape"

    pn1 = (p-1)*n
    # Reshape VAR(p) as a VAR(1)
    A1 = np.block([[A.reshape((n, n*p), order="F")],
            [np.eye(pn1), np.zeros((pn1, n))]])
    
    # Calculate the spectral norm
    rho = max(abs(np.linalg.eig(A1)[0]))  # Spectral radius
    
    if newrho is None:  # Return the spectral norm
        return rho
    else:  # Adjust and return the coefficients with the new spectral norm
        if newrho > 1: 
            warnings.warn("Selected 'newrho' is bigger than 1. The process will be unstable.")
        # Scale the coefficients
        adjusted_A = var_decay(A, newrho / rho) 
        return adjusted_A, rho



def var_rand(n, p, rho, w=None):
    """
    Generate a random VAR coefficients sequence with given spectral radius.

    Parameters:
        n (int or ndarray): Observation variable dimension or connectivity matrix/array.
        p (int): Number of lags.
        rho (float): Spectral radius.
        w (float or None): VAR coefficients decay weighting factor. Default is None (no weighting).

    Returns:
        A (ndarray): VAR coefficients sequence (3D array, last index is lag).
    """

    if isinstance(n, int):
        A = np.random.randn(n, n, p)
    else:
        C = n
        if len(C.shape)==2: C = C[:,:,np.newaxis]
        n, n1, q = C.shape
        assert n1 == n, "Connectivity matrix must be square"
        if q == 1:
            C = np.repeat(C, p, axis=2)
        else:
            if p is not None: warnings.warn("Full connectivity array was given, ignoring model order 'p'")
            p = q
        A = C * np.random.randn(n, n, p)

    if w is None:
        A = specnorm(A, rho)[0]
    else:
        A = specnorm(np.exp(-w * np.sqrt(p)) * A, rho)[0]

    return A


def var_to_autocov(A, V, q, verbose=False):
    """
    Calculate autocovariance matrices from VAR parameters.

    Parameters:
        A (ndarray): VAR coefficient matrix of shape (L, L, p), where:
                     - `L` is the number of variables in the VAR model.
                     - `p` is the number of lags.
        V (ndarray): Covariance matrix of the residuals (L x L).
        q (int): Number of desired autocovariance matrices.
        verbose (bool): If True, prints the autocovariance matrices. Default is False.

    Returns:
        Gamma (ndarray): Autocovariance matrices of shape (L, L, q, where:
                         - `Gamma[:, :, 0]` is the variance (lag 0).
                         - `Gamma[:, :, k]` for `k > 0` is the autocovariance at lag k.
    """
    # define dimensions
    if len(A.shape)==2: A = A[:,:,np.newaxis]
    L = A.shape[0]
    p = A.shape[2]
    Lp = L*p; Lp1 = L*(p-1)

    # formally rewrite the VAR(p) model as a VAR(1)
    AA = np.block([[A.reshape((L, Lp), order="F")],[np.eye(Lp1), np.zeros((Lp1, L))]])
    VV = np.block([[V, np.zeros((L, Lp1))],[np.zeros((Lp1,Lp))]])

    # solve the Lyapunov equation and get the autocovariances
    Gamma = solve_discrete_lyapunov(AA, VV)
    Gamma = Gamma[:L,:].reshape((L,L,p), order="F")
    if q<p: 
        return Gamma[:,:,:q]
    else:
        for qq in range(p,q+1):
            Gamma_qq = sum(A[:,:,l] @ Gamma[:,:,qq-l-1] for l in range(qq))
            Gamma = np.append(Gamma, Gamma_qq[:,:,np.newaxis], axis=2)

    # if desired, print the autocovariances
    if verbose:
        for idx in range(Gamma.shape[2]):
            print(f"Gamma_{idx}:\n", Gamma[:,:,idx])

    return Gamma


def var_to_covariance(A, V, p=None):
    """
    Compute the full covariance matrix of a VAR system.
    NB: Covariance matrix is ordered with past variables first, then future variables.
    
    Parameters:
        A (numpy.ndarray):
            VAR coefficients of shape (n, n, p).
        V (numpy.ndarray): 
            Residual covariance matrix of shape (n, n).
        p (int): 
            Model order.
    
    Returns:
        numpy.ndarray: 
            Covariance matrices of shape (n, n, p+1).
    """

    assert len(A.shape)==2 or len(A.shape)==3
    if len(A.shape)==2: A = A[:,:,np.newaxis]

    if p is None: p = A.shape[2]
    assert p > 0, "Enter a valid model order"
    assert A.shape[2] == p, "Model order and number of evolution matrices are not the same"
    
    L = A.shape[0]
    assert A.shape[0] == L and A.shape[1]==L, "Sources and evolution matrix dimensions are not compatible"
    assert V.shape[0] == L and V.shape[1]==L, "Sources and covariance matrix dimensions are not compatible"
    
    # Solve the Lyapunov equation
    Gamma = var_to_autocov(A, V, p)
    
    # get the full covariance of the system (past first, then future variables)
    Sigma_full = np.empty((L*(p+1),L*(p+1)))
    for s in range(p+1):
        for t in range(p+1):
            Sigma_full[s*L:(s+1)*L, t*L:(t+1)*L] = Gamma[:, :, abs(t-s)].T if t > s else Gamma[:, :, abs(t-s)]

    return Sigma_full