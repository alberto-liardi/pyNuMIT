import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import qr, cholesky
from scipy.stats import beta

from VAR.VAR_utils import specnorm


def isint(x):
    return np.all(np.equal(np.mod(x, 1), 0))


def speclim(A, M, r1, r2):
    assert specnorm(A - r1 * M) > 1 and specnorm(A - r2 * M) < 1
    while True:
        r = 0.5 * (r1 + r2)
        rho = specnorm(A - r * M)
        if rho > 1:
            r1 = r
        else:
            r2 = r
        if abs(r1 - r2) < np.finfo(float).eps:
            break
    return r


def iss_params_check(A, C, K, V=None):
    """
    Validate state-space parameters and optionally compute Cholesky factor.

    Parameters:
    A : (r, r) ndarray
    C : (n, r) ndarray
    K : (r, n) ndarray
    V : (n, n) ndarray, optional

    Returns:
    n : int
        Observation dimension
    r : int
        State dimension
    L : (n, n) ndarray or None
        Lower-triangular Cholesky factor of V (if positive-definite)
    """
    if hasattr(A, "inverse"):
        A = A.detach().numpy().copy()
        C = C.detach().numpy().copy()
        K = K.detach().numpy().copy()
        V = V.detach().numpy().copy() if V is not None else None

    A = np.atleast_2d(A)
    C = np.atleast_2d(C)
    K = np.atleast_2d(K)

    r, r1 = A.shape
    assert r1 == r, "SS: bad 'A' parameter"

    n, r1 = C.shape
    assert r1 == r, "SS: bad 'C' parameter"

    r1, n1 = K.shape
    assert n1 == n and r1 == r, "SS: bad 'K' parameter"

    L = None
    if V is not None:
        V = np.atleast_2d(V)
        n1, n2 = V.shape
        assert n1 == n and n2 == n, "SS: bad 'V' parameter"
        L = np.linalg.cholesky(V)

    if L is not None:
        return n, r, L
    else:
        return n, r


def ss_params_check(A, C, Q=None, R=None, S=None):
    """
    Validate state-space parameters for a state-space model.
    NB: This function uses python notation for C.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        State transition matrix.
    C : ndarray, shape (n, m)
        Observation matrix.
    Q : ndarray, shape (n, n), optional
        Process noise covariance matrix (default: None).
    R : ndarray, shape (m, m), optional
        Measurement noise covariance matrix (default: None).
    S : ndarray, shape (n, m), optional
        Cross-correlation matrix between process and measurement noise (default: None).
    """
    if hasattr(A, "inverse"):
        A = A.detach().numpy().copy()
        C = C.detach().numpy().copy()
        Q = Q.detach().numpy().copy() if Q is not None else None
        R = R.detach().numpy().copy() if R is not None else None
        S = S.detach().numpy().copy() if S is not None else None

    A = np.atleast_2d(A)
    C = np.atleast_2d(C)
    n, m = C.shape
    assert n > 0 and m > 0, "C must be a non-empty matrix"
    assert A.shape == (n, n), "A must be a square matrix of shape (n, n)"
    if Q is not None:
        Q = np.atleast_2d(Q)
        assert Q.shape == (n, n), "Q must be a square matrix of shape (n, n)"
    if R is not None:
        R = np.atleast_2d(R)
        assert R.shape == (m, m), "R must be a square matrix of shape (m, m)"
    if S is not None:
        S = np.atleast_2d(S)
        assert S.shape == (n, m), "S must be a matrix of shape (n, m)"

    # if Q is not None and R is not None:
    #     Sigma = np.block([[Q, S], [S.T, R]]) if S is not None else np.block([[Q, np.zeros((n, m))], [np.zeros((m, n)), R]])
    #     try:
    #         np.linalg.cholesky(Sigma)
    #     except np.linalg.LinAlgError:
    #         raise ValueError("Covariance matrix must be positive-definite, check Q, R, and S parameters.")


def iss_rand(n, m, rhoa, dis=False):
    """
    Generate random stable and minimum-phase innovations form parameters
    for a state-space model.

    Parameters
    ----------
    n : int
        Observation variable dimension.
    m : int
        State dimension (model order).
    rhoa : float
        Desired spectral norm of the state transition matrix A (must be < 1).
    dis : bool, optional
        If True, display a plot of the spectral norm of the AR transition matrix pencil.

    Returns
    -------
    A : ndarray, shape (m, m)
        State transition matrix with spectral norm equal to `rhoa`.
    C : ndarray, shape (n, m)
        Observation matrix.
    K : ndarray, shape (m, n)
        Kalman gain matrix.
    rhob : float
        Spectral norm of the AR transition matrix A - K*C.
    """
    assert rhoa < 1, "rhoa must be less than 1 for model stability."

    A = specnorm(np.random.randn(m, m), rhoa)[0][:, :, 0]
    C = np.random.randn(n, m)
    K = np.random.randn(m, n)

    M = K @ C
    rmin = speclim(A, M, -1, 0)
    rmax = speclim(A, M, +1, 0)

    r = rmin + (rmax - rmin) * np.random.rand()
    sqrtr = np.sqrt(abs(r))
    C = sqrtr * C
    K = np.sign(r) * sqrtr * K

    rhob = specnorm(A - K @ C)

    if dis:
        nr = 1000
        ramax = 1.1 * max(rmax, -rmin)
        rr = np.linspace(-ramax, ramax, nr)
        rrhob = np.array([specnorm(A - r_ * M) for r_ in rr])
        rholim = [0.9 * min(rrhob), 1.1 * max(rrhob)]

        plt.figure()
        plt.plot(rr, rrhob, label="rho(B)")
        plt.axhline(y=1, color="k", linestyle="-")
        plt.axvline(x=0, color="r", linestyle="-")
        plt.axvline(x=r, color="g", linestyle="-")
        plt.xlim([-ramax, ramax])
        plt.ylim(rholim)
        plt.xlabel("r")
        plt.ylabel("Spectral norm (rho)")
        plt.legend()
        plt.title(f"rho(A) = {rhoa:.3g}, rho(B) = {rhob:.3g}")
        plt.grid(True)
        plt.show()

    return A, C, K, rhob


def plot_svc(sval, svc, mosvc, rmax, plotm=0):
    mo = np.arange(1, rmax + 1)
    gap = 0.05
    ssvc = gap + (1 - gap) * (svc - np.min(svc)) / (np.max(svc) - np.min(svc))

    wsvc = "*" if mosvc == rmax else ""

    if plotm == 0:
        plt.figure(figsize=(8, 6))
    else:
        plt.figure(plotm)
    plt.clf()

    xlims = [0, rmax]

    # Plot SVC
    plt.subplot(2, 1, 1)
    plt.plot(mo, ssvc, "o-", label=f"SVC (opt = {mosvc}{wsvc})")
    plt.axvline(mosvc, color="r", linestyle="--")
    plt.title("Singular Value Criterion (SVC)")
    plt.xlabel("State space dimension")
    plt.ylabel("SVC (scaled)")
    plt.grid(True)
    plt.legend()
    plt.xlim(xlims)
    plt.ylim([0, 1 + gap])

    # Plot singular values
    plt.subplot(2, 1, 2)
    plt.bar(mo, sval, color=(0.65, 0.75, 1), width=1.01)
    plt.axvline(mosvc, color="r", linestyle="--")
    plt.title("Singular values")
    plt.xlabel("State space dimension")
    plt.ylabel("Singular value")
    plt.grid(True)
    plt.xlim(xlims)

    plt.suptitle(f"SS SVC model order selection (CCA, max = {rmax})", fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def onion(n, eta=1, Rmean=False):
    """
    Generate a random correlation matrix using the 'onion' method.

    Parameters
    ----------
    n : int
        Dimension of the correlation matrix.
    eta : float, optional
        Concentration parameter (default: 1, i.e., uniform).
    Rmean : bool, optional
        If True, returns the theoretical mean of the determinant |R|.

    Returns
    -------
    R : ndarray or float
        Random correlation matrix (if Rmean is False), or
        theoretical mean of |R| (if Rmean is True).
    """
    if Rmean:
        nn = np.arange(1, n)
        f = 2 * eta + nn
        return np.exp(np.sum(nn * (np.log(f - 1) - np.log(f))))

    R = np.eye(n)
    b = eta + (n - 2) / 2
    r = 2 * beta.rvs(b, b) - 1
    R[0:2, 0:2] = np.array([[1, r], [r, 1]])

    for k in range(2, n):
        b -= 0.5
        u = np.random.rand(k)
        u /= np.sqrt(np.sum(u**2))
        L = cholesky(R[:k, :k], lower=True)
        scale = np.sqrt(beta.rvs(k / 2, b))
        z = L @ (scale * u)
        R[:k, k] = z
        R[k, :k] = z

    return R


def corr_rand(n, g=None, vexp=2, tol=np.sqrt(np.finfo(float).eps), maxretries=1000):
    """
    Generate a random correlation matrix with specified multi-information.

    Parameters
    ----------
    n : int
        Number of dimensions.
    g : float or None
        Target multi-information (g = -log|R|). If None, a random correlation
        matrix is sampled using the "onion" method. If g == 0, identity matrix
        is returned. If g < 0, -g is treated as a factor applied to the mean
        multi-information from uniform sampling.
    vexp : float
        Variance exponent (default 2).
    tol : float
        Numerical tolerance (default sqrt(machine epsilon)).
    maxretries : int
        Maximum retries to find suitable matrix (default 1000).

    Returns
    -------
    R : ndarray
        Correlation matrix (n x n).
    L : ndarray
        Cholesky (left) factor such that L @ L.T = R.
    grerr : float
        Relative error in multi-information.
    retries : int
        Number of retries.
    iters : int
        Number of binary chop iterations.
    """
    retries = 0
    iters = 0
    grerr = np.nan

    if g is None:
        from scipy.stats import ortho_group

        L = ortho_group.rvs(n)
        R = L @ L.T
        return R, L, grerr, retries, iters

    if np.abs(g) < np.finfo(float).eps:
        R = np.eye(n)
        L = np.eye(n)
        return R, L, 0.0, 0, 0

    if g < 0:
        g = -g * onion(n, Rmean=True)

    gtarget = g

    for retries in range(maxretries + 1):
        Q, Rq = qr(np.random.randn(n, n))
        v = np.abs(np.random.randn(n)) ** vexp
        M = Q @ np.diag(np.sign(np.diag(Rq)))
        V = M @ np.diag(v) @ M.T
        g_est = np.sum(np.log(np.diag(V))) - np.sum(np.log(v))
        if g_est >= gtarget:
            break
    else:
        raise RuntimeError("corr_rand timed out on retries (g too large?)")

    D = np.diag(V)
    c = 1.0
    g_est = np.sum(np.log(D + c) - np.log(v + c))
    iters = 1
    while g_est > gtarget:
        c *= 2
        g_est = np.sum(np.log(D + c) - np.log(v + c))
        iters += 1

    clo, chi = 0, c
    while chi - clo > tol:
        c = (clo + chi) / 2
        g_est = np.sum(np.log(D + c) - np.log(v + c))
        iters += 1
        if g_est < gtarget:
            chi = c
        else:
            clo = c

    grerr = np.abs(gtarget - g_est) / gtarget
    V = M @ np.diag(v + c) @ M.T

    try:
        L = cholesky(V, lower=True)
    except np.linalg.LinAlgError:
        raise RuntimeError("Resulting covariance matrix not positive-definite")

    L = np.diag(1.0 / np.sqrt(np.sum(L * L, axis=1))) @ L
    R = L @ L.T
    return R, L, grerr, retries, iters


def ss_info(A, C, K, V=None, report=1):
    """
    Compute and return information about a state-space model.

    Parameters
    ----------
    A : ndarray, shape (r, r)
        State transition matrix.
    C : ndarray, shape (n, r)
        Observation matrix.
    K : ndarray, shape (r, n)
        Kalman gain matrix.
    V : ndarray, shape (n, n), optional
        Innovations covariance matrix (default: None, identity matrix assumed).
    report : int, optional
        Reporting level (default: 1). If 1, prints a summary of the model information.
        If > 1, returns a dictionary with detailed information.

    Returns
    -------
    info : dict
        Dictionary containing information about the state-space model.
    """

    def nextpow2(x):
        """Next power of 2 greater than or equal to x"""
        return int(np.ceil(np.log2(x)))

    def maxabs(X):
        return np.max(np.abs(X))

    def bitset(x, pos):
        return x | (1 << (pos - 1))

    def bitget(x, pos):
        return bool(x & (1 << (pos - 1)))

    info = {}
    r, r1 = A.shape
    assert r1 == r
    n, r1 = C.shape
    assert r1 == r
    r1, n1 = K.shape
    assert n1 == n and r1 == r

    if V is None:
        V = np.eye(n)
    else:
        n1, n2 = V.shape
        assert n1 == n and n2 == n

    info["error"] = 0
    info["observ"] = n
    info["morder"] = r

    info["rhoA"] = specnorm(A)
    info["rhoB"] = specnorm(A - K @ C)

    info["acdec"] = int(
        np.ceil(np.log(np.finfo(float).eps) / np.log(max(info["rhoA"], info["rhoB"])))
    )

    if maxabs(np.triu(V, 1) - np.triu(V.T, 1)) > np.finfo(float).eps:
        info["sigspd"] = 1  # not symmetric
    else:
        try:
            np.linalg.cholesky(V)
            info["sigspd"] = 0  # symmetric, positive definite
        except np.linalg.LinAlgError:
            info["sigspd"] = 2  # symmetric, not positive definite

    rhotol = np.sqrt(np.finfo(float).eps)

    if info["rhoA"] > 1 + rhotol:
        info["error"] = bitset(info["error"], 1)  # explosive
    elif info["rhoA"] > 1 - rhotol:
        info["error"] = bitset(info["error"], 2)  # unit root

    if info["rhoB"] > 1 + rhotol:
        info["error"] = bitset(info["error"], 3)  # explosive
    elif info["rhoB"] > 1 - rhotol:
        info["error"] = bitset(info["error"], 4)  # unit root

    if info["sigspd"] == 1:
        info["error"] = bitset(info["error"], 5)  # not symmetric
    elif info["sigspd"] == 2:
        info["error"] = bitset(info["error"], 6)  # not positive definite

    if report == 1:
        print("\nSS info:")
        print(f"    observables       = {info['observ']}")
        print(f"    model order       = {info['morder']}")
        print(f"    AR spectral norm  = {info['rhoA']:.6f}", end="")
        if bitget(info["error"], 1):
            print("ERROR: unstable (explosive)")
        elif bitget(info["error"], 2):
            print("ERROR: unstable (unit root)")
        else:
            print("stable")

        print(f"MA spectral norm  = {info['rhoB']:.6f}", end="")
        if bitget(info["error"], 3):
            print("ERROR: not minimum phase (explosive)")
        elif bitget(info["error"], 4):
            print("ERROR: not minimum phase (unit root)")
        else:
            print("minimum phase")

        print("residuals covariance matrix", end="")
        if bitget(info["error"], 5):
            print("ERROR: not symmetric")
        elif bitget(info["error"], 6):
            print("ERROR: not positive definite")
        else:
            print("       symmetric, positive definite")

        print(f"autocorr. decay   = {info['acdec']:<12}\n")

    elif report > 1:
        if info["error"] == 0:
            info["errmsg"] = ""
            return info

        info["nerrors"] = sum(bitget(info["error"], i) for i in range(1, 9))
        info["errmsg"] = "SS ERRORS" if info["nerrors"] > 1 else "SS ERROR"

        if bitget(info["error"], 1):
            info[
                "errmsg"
            ] += f": AR spectral norm = {info['rhoA']:.6f} - unstable (explosive)"
        elif bitget(info["error"], 2):
            info[
                "errmsg"
            ] += f": AR spectral norm = {info['rhoA']:.6f} - unstable (unit root)"

        if bitget(info["error"], 3):
            info[
                "errmsg"
            ] += f": MA spectral norm = {info['rhoB']:.6f} - not miniphase (explosive)"
        elif bitget(info["error"], 4):
            info[
                "errmsg"
            ] += f": MA spectral norm = {info['rhoB']:.6f} - not miniphase (unit root)"

        if bitget(info["error"], 5):
            info["errmsg"] += ": res. cov. matrix not symmetric"
        elif bitget(info["error"], 6):
            info["errmsg"] += ": res. cov. matrix not positive definite"

    return info


import numpy as np

def bandlimit(P, dim=0, fs=None, B=None):
    """
    Compute the band-limited average of a power spectrum.

    Parameters
    ----------
    P : ndarray
        Power spectrum or similar frequency-domain quantity.
    dim : int, optional
        The dimension along which frequencies vary (default: 0).
    fs : float, optional
        Sampling frequency. Default is 2π (angular frequency space).
    B : list or tuple of float, optional
        Frequency band [f_min, f_max] to average over. If None, averages over the full range [0, Nyquist].

    Returns
    -------
    PBL : ndarray
        Band-limited average of P over the specified frequency band.
    """
    if fs is None:
        fs = 2 * np.pi  # default to angular frequency

    P = np.asarray(P)
    nd = P.ndim
    assert 0 <= dim < nd, "bad dimension argument"

    N = P.shape[dim]
    fN = fs / 2.0  # Nyquist frequency

    # Determine frequency indices for integration
    if B is None:
        imin = 0
        imax = N - 1
    else:
        if not (isinstance(B, (list, tuple, np.ndarray)) and len(B) == 2):
            raise ValueError("Frequency band must be a 2-element vector [fmin, fmax]")
        B = list(B)
        fac = (N - 1) / fN
        if np.isnan(B[0]):
            B[0] = 0.0
            imin = 0
        else:
            assert 0 <= B[0] < fN, "fmin out of range"
            imin = int(round(fac * B[0]))
            imin = max(imin, 0)

        if np.isnan(B[1]):
            B[1] = fN
            imax = N - 1
        else:
            assert 0 < B[1] <= fN, "fmax out of range"
            imax = int(round(fac * B[1]))
            imax = min(imax, N - 1)

        assert B[0] < B[1], "Band must be ascending"
        assert imin < imax, "frequency band limits too close together?"

    # Move target dim to front
    axes = list(range(nd))
    if dim != 0:
        axes[0], axes[dim] = axes[dim], axes[0]
        P = np.transpose(P, axes)

    # Collapse all other dimensions
    P_flat = P.reshape((N, -1))

    # Integrate across selected frequency range
    PBL_flat = np.trapz(P_flat[imin:imax+1, :], dx=1.0, axis=0) / (N - 1)

    # Restore output shape (excluding dim)
    out_shape = [s for i, s in enumerate(P.shape) if i != 0]
    if len(out_shape) == 0:
        return PBL_flat.item()  # scalar output
    else:
        return PBL_flat.reshape(out_shape)


def ss_to_cpsd(A, C, K, V, fres, autospec=False):
    """
    Compute the cross power spectral density (CPSD) from state-space model parameters.

    Parameters
    ----------
    A : ndarray (r, r)
        State transition matrix.
    C : ndarray (n, r)
        Observation matrix.
    K : ndarray (r, n)
        Kalman gain matrix.
    V : ndarray (n, n)
        Covariance matrix of residuals (innovation noise).
    fres : int
        Frequency resolution (number of frequency points = fres + 1).
    autospec : bool, optional
        If True, return only the autospectra (diagonal elements of CPSD), transposed to shape (fres+1, n).
        Default is False.

    Returns
    -------
    S : ndarray
        Cross power spectral density. Shape:
            - (fres+1, n) if `autospec` is True (autospectra only).
            - (n, n, fres+1) otherwise.
    H : ndarray, optional
        Transfer function values at each frequency. Returned only if `autospec=False` and requested.
        Shape: (n, n, fres+1)

    Notes
    -----
    This function evaluates the CPSD over the normalized frequency interval [0, π] with `fres+1` frequency points.

    The formula used is:
        S(f) = H(f) * L * Lᵀ * H(f)ᵀ

    where L is the Cholesky factor of V such that V = L * Lᵀ,
    and H(f) = I + C * ((e^{iπf} I - A)⁻¹ * K)

    This version assumes a stable SS model and that V is symmetric positive definite.
    """

    n, r, L = iss_params_check(A, C, K, V)
    h = fres + 1
    In = np.eye(n)
    Ir = np.eye(r)
    S = np.zeros((n, n, h), dtype=complex)
    w = np.exp(1j * np.pi * np.arange(h) / fres)

    # Optionally store the transfer function H(f)
    H_out = np.zeros((n, n, h), dtype=complex) if not autospec else None

    for k in range(h):
        Hk = In + C @ np.linalg.solve(w[k] * Ir - A, K)
        HLk = Hk @ L
        S[:, :, k] = HLk @ HLk.conj().T
        if H_out is not None:
            H_out[:, :, k] = Hk

    if autospec:
        # Return only diagonal (autospectra), transpose to match MATLAB shape [fres+1, n]
        s = np.array([np.diag(S[:, :, k]) for k in range(h)])
        return s.real, None  # transpose not needed, already [fres+1, n]
    else:
        return (S.real, H_out) if H_out is not None else S.real


def whiten_noise(A, C, Q, R, S):
    """
    Whiten the noise in a state-space model by transforming the system matrices.

    Parameters:
    A (torch.Tensor or np.ndarray): State transition matrix. Shape (n, n).
    C (torch.Tensor or np.ndarray): Observation matrix. Shape (n, m).
    Q (torch.Tensor or np.ndarray): Process noise covariance matrix. Shape (n, n).
    R (torch.Tensor or np.ndarray): Measurement noise covariance matrix. Shape (m, m).
    S (torch.Tensor or np.ndarray): Cross-covariance matrix between process and measurement noise. Shape (n, m).

    Returns:
    A_bar, C_bar, Q_bar, R_bar: Transformed system matrices with whitened noise.
    """
    ss_params_check(A, C, Q, R, S)

    if hasattr(R, "inverse"):
        import torch

        R_inv = torch.linalg.inv(R)
    else:
        R_inv = np.linalg.inv(R)
    A_bar = A - C @ R_inv @ S.T
    Q_bar = Q - S @ R_inv @ S.T
    C_bar = C
    R_bar = R

    return A_bar, C_bar, Q_bar, R_bar


from scipy.linalg import solve_discrete_are, inv


def mdare(A, C, Q, R, S, method="scipy"):
    """
    Solve DARE for associated State Space Model.

    Parameters:
    A (torch.Tensor or ndarray): State transition matrix. Shape (n, n).
    C (torch.Tensor or ndarray): Observation matrix. Shape (n, m).
    Q (torch.Tensor or ndarray): Process noise covariance matrix. Shape (n, n).
    R (torch.Tensor or ndarray): Measurement noise covariance matrix. Shape (m, m).
    S (torch.Tensor or ndarray): Cross-covariance matrix between process and measurement noise. Shape (n, m).
    """
    ss_params_check(A, C, Q, R, S)

    if method == "scipy":
        P = solve_discrete_are(A, C, Q, R, s=S)
        V = C.T @ P @ C + R
        K = (A.T @ P @ C + S) @ inv(V)
    elif method == "torch":
        import torch

        # make sure that 'dare-torch' is in path
        try:
            from riccati import dare
        except ImportError:
            raise ImportError(
                "Please install 'dare-torch' package and add it to the path to use the 'torch' method."
            )

        # dare below does not support cross-diagonal term S, so we need to whiten the noise first
        A_w, C_w, Q_w, R_w = whiten_noise(A, C, Q, R, S)

        if isinstance(A_w, np.ndarray):
            A_t = torch.tensor(A_w, dtype=torch.float64, requires_grad=False)
            C_t = torch.tensor(C_w, dtype=torch.float64, requires_grad=False)
            Q_t = torch.tensor(Q_w, dtype=torch.float64, requires_grad=False)
            R_t = torch.tensor(R_w, dtype=torch.float64, requires_grad=False)
        else:
            A_t = A_w
            C_t = C_w
            Q_t = Q_w
            R_t = R_w

        dare_solver = dare()
        P_t = dare_solver(A_t, C_t, Q_t, R_t)
        V_t = C_t.T @ P_t @ C_t + R_t
        # P and V are invariant under whitening, but K is not. We need to use the original matrices:
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float64, requires_grad=False).copy()
            S = torch.tensor(S, dtype=torch.float64, requires_grad=False).copy()
        K_t = (A.T @ P_t @ C_t + S) @ torch.linalg.inv(V_t)

        if isinstance(A_w, np.ndarray):
            P = P_t.detach().numpy()
            V = V_t.detach().numpy()
            K = K_t.detach().numpy()
        else:
            P = P_t
            V = V_t
            K = K_t
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    return K, V, P
