import numpy as np
from misc import h, RedMMI, RedCCS

def PID_calculator(Sigma, S1=1, S2=1, T=1, red_fun="MMI", as_dict=False, verbose=False, p=None, **kwargs):
    """
    Calculate Partial Information Decomposition (PID) components from a full covariance matrix.
    Supports both VAR(p) and Gaussian systems.
    NB: covariance matrix should be ordered with past variables first, then future variables.

    Parameters:
        Sigma (np.ndarray): Full covariance matrix with sources S=S1+S2 and target T.
                            Shape: ((S*p+T), (S*p+T)) for VAR(p), or (S+T, S+T) for Gaussian (p=1).
                            Assumes variable order: [past (S*p)], [target (T)].
        S1 (int): Dimension of the first source group (default is 1).
        S2 (int): Dimension of the second source group (default is 1).
        T (int): Number of target variables (default is 1).
        red_fun (str, optional): Redundancy function to use (default is "MMI").
        as_dict (bool): If True, return result as a dictionary. Otherwise, return as 1D array.
        verbose (bool): Verbose output.
        **kwargs: Additional keyword arguments for compatibility.

    Returns:
        PID (dict or np.ndarray): Dictionary or array with [Redundancy, Unique_X, Unique_Y, Synergy]
    """

    # Determine source dimensions
    assert Sigma.shape[0] == Sigma.shape[1], "Covariance matrix must be square."
    S = S1 + S2

    total_dim = Sigma.shape[0]
    assert total_dim >= S + T, "Covariance matrix must have at least S+T variables." 

    # number of past lags
    if p is None: p = (total_dim - T) // S 
    assert (isinstance(p, int) or np.isscalar(p)) and p >= 0, "Invalid past lags. Check the dimension of your covariance matrix."

    # Split Sigma
    Sigma_S = Sigma[:S*p, :S*p]
    Sigma_T = Sigma[S*p:, S*p:]
    Sigma_ST = Sigma[:S*p, S*p:]

    # Entropies and MI
    H_T = h(Sigma_T)
    H_S = h(Sigma_S)
    H_TS = h(Sigma)

    MI = (H_T + H_S - H_TS) / np.log(2)

    # Build S1 and S2 pasts: interleaved ordering due to lags 
    idx_S1 = np.concatenate([np.arange(i*S, i*S + S1) for i in range(p)])
    idx_S2 = np.concatenate([np.arange(i*S + S1, (i+1)*S) for i in range(p)])

    Sigma_S1 = Sigma_S[np.ix_(idx_S1, idx_S1)]
    Sigma_S2 = Sigma_S[np.ix_(idx_S2, idx_S2)]

    Sigma_TS1 = np.block([
        [Sigma_S1, Sigma_ST[idx_S1, :]],
        [Sigma_ST[idx_S1, :].T, Sigma_T]
    ])
    Sigma_TS2 = np.block([
        [Sigma_S2, Sigma_ST[idx_S2, :]],
        [Sigma_ST[idx_S2, :].T, Sigma_T]
    ])

    H_TS1 = h(Sigma_TS1)
    H_TS2 = h(Sigma_TS2)

    MI_S1 = (H_T + h(Sigma_S1) - H_TS1) / np.log(2)
    MI_S2 = (H_T + h(Sigma_S2) - H_TS2) / np.log(2)

    # Calculate redundancy
    if red_fun == "MMI":
        Red = RedMMI(MI_S1, MI_S2)
    elif red_fun == "CCS":
        Red = RedCCS(MI_S1, MI_S2, MI)
    elif red_fun == "Broja":
        from gpid.tilde_pid import exact_gauss_tilde_pid

        # need to set the future first and the past second
        idx_T = list(np.arange(S*p, total_dim))
        idx_S1 = list(idx_S1)
        idx_S2 = list(idx_S2)
        # print(idx_T + idx_S1 + idx_S2)
        _, _, Red, _ = exact_gauss_tilde_pid(
                            Sigma[np.ix_(idx_T + idx_S1 + idx_S2, idx_T + idx_S1 + idx_S2)], 
                            dm=T, dx=S1*p, dy=S2*p)[-4:]
    else:
        raise NotImplementedError
    
    UnX = MI_S1 - Red
    UnY = MI_S2 - Red
    Syn = MI - (Red + UnX + UnY)

    if as_dict:
        return {"Red": Red, "UnX": UnX, "UnY": UnY, "Syn": Syn}
    else:
        return np.array([Red, UnX, UnY, Syn])