import numpy as np
    
from InfoDyn.PID import PID_calculator
from InfoDyn.PhiID import PhiID_calculator
from misc import h, conditional_cov


def mutual_information(cov, idx1, idx2):
    """
    Compute mutual information between two sets of variables.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        idx1 (list): Indices of the first set of variables.
        idx2 (list): Indices of the second set of variables.
    
    Returns:
        float: Mutual information value in BITS.
    """
    if isinstance(idx1, int) or np.isscalar(idx1):
        idx1 = [idx1]
    if isinstance(idx2, int) or np.isscalar(idx2):
        idx2 = [idx2]
    full_cov = cov[np.ix_(idx1 + idx2, idx1 + idx2)]
    cov1 = cov[np.ix_(idx1, idx1)]
    cov2 = cov[np.ix_(idx2, idx2)]
    return (h(cov1) + h(cov2) - h(full_cov)) / np.log(2)


def transfer_entropy(Sigma, idx_T_future, idx_T_past, idx_S_past):
    """
    Compute Transfer Entropy TE_{S -> T} from covariance matrix.

    Parameters
        Sigma (ndarray): Covariance matrix of shape (d, d), where d = total variables.
        idx_T_future (list): Indices for T_{t+1}.
        idx_T_past (list): Indices for T_t.
        idx_S_past (list): Indices for S_t.

    Returns
        TE (float): Transfer entropy from S to T in nats.
    """
    # Conditional covariance with only T_past
    cov_T_future_given_T_past = conditional_cov(Sigma, idx_T_future, idx_T_past)

    # Conditional covariance with T_past and S_past
    cov_T_future_given_T_and_S_past = conditional_cov(
                                    Sigma, idx_T_future, idx_T_past + idx_S_past)

    # Calculate the conditional entropies
    H_TT  = h(cov_T_future_given_T_past)
    H_TST = h(cov_T_future_given_T_and_S_past)

    TE = H_TT - H_TST
    return TE / np.log(2)


def pearson_correlation(cov, idx1, idx2):
    """
    Compute the Pearson correlation coefficient between two variables.

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        idx1 (list): Indices of the first variable.
        idx2 (list): Indices of the second variable.

    Returns:
        float: Pearson correlation coefficient.
    """
    # Ensure idx1 and idx2 are lists of length 1
    if isinstance(idx1, int) or np.isscalar(idx1):
        idx1 = [idx1]
    if isinstance(idx2, int) or np.isscalar(idx2):
        idx2 = [idx2]
    assert len(idx1) == 1 and len(idx2) == 1, "Pearson correlation requires single indices."

    var1 = cov[idx1, idx1][0]
    var2 = cov[idx2, idx2][0]
    covar = cov[idx1, idx2][0]
    return covar / np.sqrt(var1 * var2)


def PID(cov, idx_p1, idx_p2, idx_t, **kwargs):
    """
    Compute the Partial Information Decomposition (PID) of a system (wrapper).

    Parameters:
        cov (numpy.ndarray): Covariance matrix.
        idx_p1 (list): Indices of the first set of source variables.
        idx_p2 (list): Indices of the second set of source variables.
        idx_t (list): Indices of the target variable(s).
        kwargs: Additional arguments for PID calculation (see InfoDyn/PID.py).

    Returns:
        np.array or dict: PID values (default: np.array).
    """
    if isinstance(idx_p1, int) or np.isscalar(idx_p1):
        idx_p1 = [idx_p1]
    if isinstance(idx_p2, int) or np.isscalar(idx_p2):
        idx_p2 = [idx_p2]
    if isinstance(idx_t, int) or np.isscalar(idx_t):
        idx_t = [idx_t]

    red_cov = cov[np.ix_(idx_p1 + idx_p2 + idx_t, idx_p1 + idx_p2 + idx_t)]
    return PID_calculator(red_cov, S1=len(idx_p1), S2=len(idx_p2), T=len(idx_t), **kwargs)


def PhiID(cov, idx_p1, idx_p2, idx_t1, idx_t2, **kwargs):
    """
    Compute the PhiID of a system (wrapper).

    Parameters: 
        cov (numpy.ndarray): Covariance matrix.
        idx_p1 (list): Indices of the first set of source variables.
        idx_p2 (list): Indices of the second set of source variables.
        idx_t1 (list): Indices of the first set of target variables.
        idx_t2 (list): Indices of the second set of target variables.
        kwargs: Additional arguments for PhiID calculation (see InfoDyn/PhiID.py).

    Returns:
        np.array or dict: PhiID values (default: np.array).
    """
    if isinstance(idx_p1, int) or np.isscalar(idx_p1):
        idx_p1 = [idx_p1]
    if isinstance(idx_p2, int) or np.isscalar(idx_p2):
        idx_p2 = [idx_p2]
    if isinstance(idx_t1, int) or np.isscalar(idx_t1):
        idx_t1 = [idx_t1]
    if isinstance(idx_t2, int) or np.isscalar(idx_t2):
        idx_t2 = [idx_t2]

    sources = idx_p1 + idx_p2
    targets = idx_t1 + idx_t2
    red_cov = cov[np.ix_(sources + targets, sources + targets)]
    return PhiID_calculator(red_cov, L1=len(idx_p1), L2=len(idx_p2), **kwargs)