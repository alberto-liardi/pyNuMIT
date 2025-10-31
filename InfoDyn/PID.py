import numpy as np
from misc import h, RedMMI, RedCCS


def PID_calculator(
    Sigma,
    S1=1,
    S2=1,
    T=1,
    red_fun="MMI",
    as_dict=False,
    verbose=False,
    p=None,
    **kwargs
):
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
    if p is None:
        p = (total_dim - T) // S
    assert (
        isinstance(p, int) or np.isscalar(p)
    ) and p >= 0, "Invalid past lags. Check the dimension of your covariance matrix."

    # Split Sigma
    Sigma_S = Sigma[: S * p, : S * p]
    Sigma_T = Sigma[S * p :, S * p :]
    Sigma_ST = Sigma[: S * p, S * p :]

    # Entropies and MI
    H_T = h(Sigma_T)
    H_S = h(Sigma_S)
    H_TS = h(Sigma)

    MI = (H_T + H_S - H_TS) / np.log(2)

    # Build S1 and S2 pasts: interleaved ordering due to lags
    idx_S1 = np.concatenate([np.arange(i * S, i * S + S1) for i in range(p)])
    idx_S2 = np.concatenate([np.arange(i * S + S1, (i + 1) * S) for i in range(p)])

    Sigma_S1 = Sigma_S[np.ix_(idx_S1, idx_S1)]
    Sigma_S2 = Sigma_S[np.ix_(idx_S2, idx_S2)]

    Sigma_TS1 = np.block(
        [[Sigma_S1, Sigma_ST[idx_S1, :]], [Sigma_ST[idx_S1, :].T, Sigma_T]]
    )
    Sigma_TS2 = np.block(
        [[Sigma_S2, Sigma_ST[idx_S2, :]], [Sigma_ST[idx_S2, :].T, Sigma_T]]
    )

    H_TS1 = h(Sigma_TS1)
    H_TS2 = h(Sigma_TS2)

    MI_S1 = (H_T + h(Sigma_S1) - H_TS1) / np.log(2)
    MI_S2 = (H_T + h(Sigma_S2) - H_TS2) / np.log(2)

    # Calculate redundancy
    if red_fun == "MMI":
        Red = RedMMI(MI_S1, MI_S2)
    elif red_fun == "CCS":
        Red = RedCCS(MI_S1, MI_S2, MI)
    elif red_fun == "Broja" or red_fun == "Delta":
        if red_fun == "Broja":
            from gpid.tilde_pid import exact_gauss_tilde_pid

            calc = exact_gauss_tilde_pid
        else:
            from gpid.estimate import approx_pid_from_cov

            calc = approx_pid_from_cov

        # need to set the future first and the past second
        idx_T = list(np.arange(S * p, total_dim))
        idx_S1 = list(idx_S1)
        idx_S2 = list(idx_S2)
        _, _, Red, _ = calc(
            Sigma[np.ix_(idx_T + idx_S1 + idx_S2, idx_T + idx_S1 + idx_S2)],
            dm=T,
            dx=S1 * p,
            dy=S2 * p,
        )[-4:]
    elif red_fun == "IG":
        pids = PID_IG(Sigma, S1 * p, S2 * p, T)
        Red = pids["pid"][0]
    else:
        raise NotImplementedError

    UnX = MI_S1 - Red
    UnY = MI_S2 - Red
    Syn = MI - (Red + UnX + UnY)

    if as_dict:
        return {"Red": Red, "UnX": UnX, "UnY": UnY, "Syn": Syn}
    else:
        return np.array([Red, UnX, UnY, Syn])


from scipy.linalg import cholesky, solve
from scipy.optimize import root_scalar, minimize


def PID_IG(Sigma, S1, S2, T):

    # Convert to correlation matrix
    d = np.diag(1 / np.sqrt(np.diag(Sigma)))
    Sigma = d @ Sigma @ d

    n0, n1, n2 = S1, S2, T
    ind0 = np.arange(n0)
    ind1 = np.arange(n0, n0 + n1)
    ind2 = np.arange(n0 + n1, n0 + n1 + n2)

    I0 = np.eye(n0)
    I1 = np.eye(n1)
    I2 = np.eye(n2)

    S00 = Sigma[np.ix_(ind0, ind0)]
    S01 = Sigma[np.ix_(ind0, ind1)]
    S02 = Sigma[np.ix_(ind0, ind2)]
    S11 = Sigma[np.ix_(ind1, ind1)]
    S12 = Sigma[np.ix_(ind1, ind2)]
    S22 = Sigma[np.ix_(ind2, ind2)]

    InvSq00 = solve(cholesky(S00, lower=False), I0)
    InvSq11 = solve(cholesky(S11, lower=False), I1)
    InvSq22 = solve(cholesky(S22, lower=False), I2)

    P = InvSq00.T @ S01 @ InvSq11
    Q = InvSq00.T @ S02 @ InvSq22
    R = InvSq11.T @ S12 @ InvSq22

    P1 = P.T
    Q1 = Q.T
    R1 = R.T

    dP = np.linalg.det(I1 - P1 @ P)
    dQ = np.linalg.det(I2 - Q1 @ Q)
    dR = np.linalg.det(I2 - R1 @ R)
    dQR = np.linalg.det(I1 - R @ Q1 @ Q @ R1)

    # Full covariance matrix
    r1 = np.hstack([I0, P, Q])
    r2 = np.hstack([P1, I1, R])
    r3 = np.hstack([Q1, R1, I2])
    Sig = np.vstack([r1, r2, r3])

    # Check positive definiteness
    ev = np.linalg.eigvals(Sig)
    if not np.all(ev > 0):
        raise ValueError("Covariance matrix is not positive definite")

    PD = "yes"

    # Compose Sig5
    R5 = P.T @ Q
    r51 = np.hstack([I0, P, Q])
    r52 = np.hstack([P1, I1, R5])
    r53 = np.hstack([Q1, R5.T, I2])
    Sig5 = np.vstack([r51, r52, r53])

    # Compose Sig6
    Q6 = P @ R
    r61 = np.hstack([I0, P, Q6])
    r62 = np.hstack([P1, I1, R])
    r63 = np.hstack([Q6.T, R1, I2])
    Sig6 = np.vstack([r61, r62, r63])

    sig5_inv = np.linalg.inv(Sig5)
    sig6_inv = np.linalg.inv(Sig6)

    # Feasibility function for t
    def feas_test(t):
        m = (1 - t) * sig5_inv + t * sig6_inv
        return 2 * int(np.all(np.linalg.eigvals(m) > 0)) - 1

    root_hi = root_scalar(feas_test, bracket=[1, 100]).root
    root_lo = root_scalar(feas_test, bracket=[-100, 0]).root
    feas = np.array([root_lo, root_hi])

    # KL divergence function
    def KLdiv(t):
        mm = np.linalg.inv((1 - t) * sig5_inv + t * sig6_inv)
        return 0.5 * (
            np.sum(np.log(np.linalg.eigvals(mm)))
            - np.sum(np.log(np.linalg.eigvals(Sig)))
        )

    # Initial t for optimization
    x0 = (root_lo + root_hi) / 2
    res = minimize(
        lambda t: KLdiv(t), x0, bounds=[(root_lo, root_hi)], method="L-BFGS-B"
    )

    tstar = res.x[0]
    syn = res.fun

    # Mutual informations
    i13 = 0.5 * np.log(1 / dQ)
    i23 = 0.5 * np.log(1 / dR)
    i13G2 = 0.5 * np.log(dP * dR / np.linalg.det(Sig))
    i23G1 = 0.5 * np.log(dP * dQ / np.linalg.det(Sig))
    jmi = 0.5 * np.log(dP / np.linalg.det(Sig))
    ii = jmi - i13 - i23

    inf = np.array([i13, i23, i13G2, i23G1, jmi, ii]) / np.log(2)

    unq1 = i13G2 - syn
    unq2 = i23G1 - syn
    red = i13 - unq1
    pid = np.array([red, unq1, unq2, syn]) / np.log(2)

    return {"PD": PD, "feas": feas, "t_star": tstar, "inf": inf, "pid": pid}
