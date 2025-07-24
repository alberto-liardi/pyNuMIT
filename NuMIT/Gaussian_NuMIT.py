import numpy as np
from scipy.stats import wishart
from scipy.optimize import fsolve

# TODO: implement Broja PID (use gpid package)
# TODO: PhiID gaussian fitness procedure

def Gauss_from_MI(MI, S=2, T=1, A=None, Sigma_s=None, Sigma_u=None, varA=1, Atype="gauss", baseU="id"):
    """
    Generate a random Gaussian system with a specified total Mutual Information (MI) between sources and targets.
    NB: mutual information is in BITS.

    Parameters:
    - MI: Desired total mutual information between sources and targets (scalar).
    - S: Number of sources in the system (default is 2).
    - T: Number of targets in the system (default is 1).
    - A: [T, S] matrix of linear coefficients relating sources to targets (optional).
    - Sigma_s: [S, S] covariance matrix of the sources (optional).
    - Sigma_u: [T, T] conditional covariance matrix of the targets given the sources (optional).
    - varA: Variance of the coefficients in A (default is 1).
    - base: Base for the covariance matrix of the targets, either "id" (identity) or "rand" (random Wishart) (default is "id").

    Returns:
    - Sigma: [S+T, S+T] covariance matrix of the full system.
    - success: Scalar flag indicating success (1) or failure (0) of the mutual information optimization.
    - A: [T, S] matrix of linear coefficients used in the system.
    - g: Scalar scaling factor applied to the conditional covariance matrix.
    """
    x0 = [0, -1, 2, -2, 5, -5]
    success = 0

    if Sigma_s is None:
        Sigma_s = wishart.rvs(df=S+1, scale=np.eye(S))
    if A is None:
        if Atype=="gauss":
            A = np.random.normal(0, varA, size=(T, S))
        elif Atype=="unif":
            if varA<0: varA = -varA
            A = np.random.uniform(-varA, varA, size=(T, S))
        else:
            raise ValueError("A type must be either 'gauss' or 'unif'.")
    if Sigma_u is None:
        if baseU == "id":
            Sigma_u = np.atleast_2d(wishart.rvs(df=T+1, scale=np.eye(T)))
        elif baseU == "rand":
            W = wishart.rvs(df=T+1, scale=np.eye(T))
            Sigma_u = np.atleast_2d(wishart.rvs(df=T+1, scale=W))
        else:
            raise ValueError("Base must be either 'id' or 'rand'.")

    def fun(x):
        # NB: mutual information is in BITS
        return np.linalg.det(np.eye(T) + x * np.linalg.solve(Sigma_u, A @ Sigma_s @ A.T)) - 2**(2 * MI)

    alpha = None
    for x in x0:
        alpha, _, success, _ = fsolve(fun, x, full_output=1)
        if success: break
    else: 
        print("Optimization failed")
        return None, None, success, None
    
    g = 1 / alpha
    Sigma_t = A @ Sigma_s @ A.T + g * Sigma_u
    Sigma_cross = A @ Sigma_s
    Sigma = np.block([[Sigma_s, Sigma_cross.T], [Sigma_cross, Sigma_t]])

    return Sigma, success, A, g


def Gauss_nulls(MIs, kind, params, verbose=False, As=None, parallel=False, n_jobs=4, **kwargs):
    """
    Generates null models for a given set of mutual information (MI) values using Gaussian processes. 

    Parameters
    ----------
    MIs : list or numpy.ndarray
        Array of mutual information (MI) values for which null models are generated.
        Each entry corresponds to a different MI value.
    kind : str
        Type of decomposition to be performed ("PID", PhiID")
    params : dict
        Configuration parameters for null model generation. Must include the following keys:
        - 'n_runs' : int
            Number of null model simulations to generate. Defaults to 1000.
        - 'S' : int
            Dimension of the Gaussian sources. Defaults to 2.
            If odd, additional keys 'S1' and 'S2' must be provided.
        - 'T' : int
            Dimension of the Gaussian target. Defaults to 1.
        - 'red_fun' : str
            Redundancy function to use. Defaults to "MMI".

    verbose : bool, optional
        If True, prints progress and warnings. Defaults to False.

    Returns
    -------
    nulls : numpy.ndarray
        A 3D array of shape (n_atoms, len(MIs), params["n_runs"]) containing the null models
        for each MI value.
    """

    # input check
    if 'n_runs' not in params:
        params['n_runs'] = 1000
    if 'S' not in params:
        params['S'] = 2
    if 'T' not in params:
        if kind=="PID":
            params['T'] = 1
        elif kind=="PhiID":
            params['T'] = params['S']
    if params['S'] % 2 != 0:
        assert 'S1' in params and 'S2' in params, \
            "Dimension of S is odd, enter the dimensions of the two PID sources explicitly."
    if 'red_fun' not in params:
        params['red_fun'] = "MMI"
    if not As:
        As = [None]*len(MIs)

    if kind=="PID":
        from InfoDyn.PID import PID_calculator
        atom_calculator = PID_calculator
        n_atoms = 4
    elif kind=="PhiID":
        from InfoDyn.PhiID import PhiID_calculator
        atom_calculator = PhiID_calculator
        n_atoms = 16
    else: 
        raise NotImplementedError
    
    def nulls_from_MI(mi, verbose, params, As=As, **kwargs):
        nerr_loc = 0
        mi_nulls = np.full((n_atoms, params["n_runs"]), np.nan)

        for n in range(params["n_runs"]):
            if n % (params["n_runs"] // 10) == 0 and verbose:
                print(f"{n / params['n_runs'] * 100}% null models completed!") 

            Sigma, succ, _, _ = Gauss_from_MI(mi, params["S"], params["T"], **kwargs)
            if not succ:
                if verbose: print("Optimisation failed, skipping...")
                nerr_loc += 1
                continue

            T = int(Sigma.shape[0] - params["S"])
            mi_nulls[:, n] = atom_calculator(Sigma, params["S"]//2, params["S"]//2, T=T, 
                                            red_fun=params["red_fun"], p=1)

        return mi_nulls, nerr_loc

    nerr = 0
    nulls = np.full((n_atoms, len(MIs), params["n_runs"]), np.nan)
    if parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(nulls_from_MI)(mi, False, params, As, **kwargs)
            for mi in MIs
        )
        for m, (nulls_res, nerr_res) in enumerate(results):
            nulls[:, m, :] = nulls_res
            nerr += nerr_res
        if verbose: print("All null models completed!")
    else:
        for m, mi in enumerate(MIs):
            if verbose: print(f"Generating null models for set {m+1} of {len(MIs)}.")
            nulls[:, m, :], mi_nerr = nulls_from_MI(mi, verbose, params, As, **kwargs)
            nerr += mi_nerr
            if verbose: print(f"100% nulls models completed!\n")

    if verbose: print(f"{nerr} optimisations have failed!\n")

    return nulls


if __name__ == "__main__":
    MI = 4.45455
    Sigma, success, A, g = Gauss_from_MI(MI, S=2, T=2)

    # I_check = h(Sigma_t) + h(Sigma_s) - h(Sigma)
    # print(I_check)
    # if abs(MI - I_check) > 1e-6:
    #     success = 1
    #     print("Error in the optimization of the MI.")

    # Sigma = np.array([[ 1.02742114, -1.33018592,  0.50852778],
    #     [-1.33018592,  1.74101346, -0.67543034],
    #     [ 0.50852778, -0.67543034,  0.26715794]])

    # X = np.random.rand(4,4)
    # Sigma = X@X.T
    from InfoDyn.PID import PID_calculator
    from InfoDyn.PhiID import PhiID_calculator
    
    print(Sigma.shape, A.shape)
    pid = PID_calculator(Sigma, S1=1, S2=1, T=2, red_fun='MMI')
    print(f"PID atoms: {pid}, sum: {np.sum(pid)}")
    phiid = PhiID_calculator(Sigma, red_fun='Broja')
    print(f"PhiID atom: {phiid}), sum: {np.sum(phiid)}")