import numpy as np
from scipy.stats import wishart
from scipy.optimize import fsolve

from VAR.VAR_utils import *
from misc import Sigmoid


def VAR_from_MI(MI, p, S, A=None, V=None, baseU="id"):
    """
    Generate a random VAR system with specified total mutual information (MI).

    Parameters:
        MI (float): Target total mutual information between sources and targets.
        p (int): Order of the VAR(p) model.
        S (int): Number of sources in the VAR system.
        A (numpy ndarray, optional): Starting coefficient matrix of the VAR system [shape (S, S, p)].
        V (numpy ndarray, optional): Starting noise covariance matrix (Wishart-distributed) [shape (S, S)].
        baseU (str, optional): Base for the noise covariance matrix, either "id" (identity) or "rand" (random Wishart). Defaults to "id".

    Returns:
        A (numpy ndarray): Coefficient matrix of the VAR system with shape (S, S, p). 
        V (numpy ndarray): Noise covariance matrix (Wishart-distributed) of shape (S, S).
        success (int): Flag indicating the success (1) or failure (0) of the optimization process.
        sr (float): Optimized spectral radius for the system.
    """
    # Initialize variables
    x0 = [0, -1, 2, -2, 5, -5]  # Initial guesses for optimization

    # Generate random VAR coefficients
    if A is None:
        A = var_rand(S, p, np.random.rand())
        while np.any(np.abs(A) < 1e-18):
            A = var_rand(S, p, np.random.rand())
    else:
        if len(A.shape)==2: A = A[:,:,np.newaxis]
        assert A.shape[2] == p, "Model order and number of evolution matrices are not the same"
        assert A.shape[0] == S and A.shape[1]==S, "Sources and evolution matrix dimensions are not compatible"

    # Generate noise covariance matrix from Wishart distribution
    if V is None:
        if baseU == "id":
            V = wishart.rvs(df=S + 1, scale=np.eye(S))
        elif baseU == "rand":
            W = wishart.rvs(df=S + 1, scale=np.eye(S))
            V = wishart.rvs(df=S + 1, scale=W)

    # Optimize to find the sectral radius `g` such that mutual information is `MI`
    def fun(x):
        x = Sigmoid(x, 1)
        B = specnorm(A, x)[0]
        try:
            G = var_to_autocov(B, V, 1)
            MI_value = (0.5 * np.linalg.slogdet(G[:, :, 0])[1] - 0.5 * np.linalg.slogdet(V)[1]) / np.log(2)
            if not np.isreal(MI_value):
                raise ValueError("Error in optimizing g: MI became complex!")
            return MI_value - MI
        except:
            return np.nan

    for x in x0:
        g, _, success, _ = fsolve(fun, x, full_output=1)
        if success: break
    else:
        print("Optimization failed")
        return None, None, success, None

    # Apply the optimized spectral radius and return the coefficients
    sr = Sigmoid(g, 1)
    A = specnorm(A, sr)[0]

    return A, V, success, sr


def VAR_nulls(MIs, kind, params, verbose=False, As=None, parallel=False, n_jobs=4, **kwargs):
    """
    Generates null models for a given set of mutual information (MI) values using Vector Autoregressive (VAR) processes. 

    Parameters
    ----------
    MIs : list or numpy.ndarray
        Array of mutual information (MI) values for which null models are generated.
        Each entry corresponds to a different MI value.
    kind : str
        Type of decomposition to be performed ("PID", PhiID")
    params : dict
        Configuration parameters for null model generation. Must include the following keys:
        - 'p' : list or numpy.ndarray
            List of model orders for each MI. If not provided, defaults to 1 for all MIs.
        - 'n_runs' : int
            Number of null model simulations to generate. Defaults to 1000.
        - 'S' : int
            Dimension of the VAR process. Defaults to 2.
            If odd, additional keys 'S1' and 'S2' must be provided.
        - 'red_fun' : str
            Redundancy function to use. Defaults to "MMI".
    verbose : bool, optional
        If True, prints progress and warnings. Defaults to False.
    As : list of numpy.ndarray, optional
        List of VAR coefficients for the null generation. If not provided, they are randomly sampled.
    parallel : bool, optional
        If True, generates null models in parallel using joblib. Defaults to False.
    n_jobs : int, optional
        Number of parallel jobs to run if `parallel` is True. Defaults to 4.
    **kwargs : additional keyword arguments

    Returns
    -------
    nulls : numpy.ndarray
        A 3D array of shape (n_atoms, len(MIs), params["n_runs"]) containing the null models
        for each MI value.
    """
    # input check
    if 'p' not in params:
        params['p'] = np.ones(len(MIs), dtype=int)  
    if 'n_runs' not in params:
        params['n_runs'] = 1000
    if 'S' not in params:
        params['S'] = 2
    if params['S'] % 2 != 0:
        assert 'S1' in params and 'S2' in params, \
            "Dimension of the VAR is odd, enter the dimensions of the two PID sources explicitly."
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
    
    def nulls_from_MI(mi, p, verbose, params=params, As=As):
        if mi is None or np.isnan(mi) or mi<1e-5: 
            if verbose: print(f"MI {mi} not valid, skipping")
            return np.full((n_atoms, params["n_runs"]),np.nan), 0
        nerr_loc = 0
        mi_nulls = np.full((n_atoms, params["n_runs"]),np.nan)
        for n in range(params["n_runs"]):
            if n%(params["n_runs"]//10)==0 and verbose: print(f"{n/params['n_runs']*100}% null models completed!") 

            A = np.ones((params["S"], params["S"], p))# Initialize A with array of ones
            while np.any(np.abs(np.linalg.eigvals(A[:, :, 0])) > 0.98):
                A,V,succ,_ = VAR_from_MI(mi, p, params["S"], As[np.random.randint(0,len(As))])
            if not succ:
                if verbose: print("Optimisation failed, skipping...")
                nerr_loc+=1
                continue

            cov = var_to_covariance(A, V, p)
            corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
            mi_nulls[:,n] = atom_calculator(corr, params["S"]//2, params["S"]//2, T=params["S"], 
                                            red_fun=params["red_fun"], p=p, verbose=verbose, as_dict=False, **kwargs).T
        return mi_nulls, nerr_loc

    nerr=0
    nulls = np.full((n_atoms,len(MIs),params["n_runs"]),np.nan)
    if parallel:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(nulls_from_MI)(mi, params["p"][m], False, 
                                                                 params, As) for m,mi in enumerate(MIs))
        for m, (nulls_res, nerr_res) in enumerate(results):
            nulls[:,m,:] = nulls_res
            nerr += nerr_res
        if verbose: print("All null models completed!")
    else:
        for m,mi in enumerate(MIs):
            if verbose: print(f"Generating null models for set {m+1} of {len(MIs)}.")
            nulls[:,m,:], mi_nerr = nulls_from_MI(mi, params["p"][m], verbose, params=params, As=As)
            nerr += mi_nerr
            if verbose: print(f"100% nulls models completed!\n")

    if verbose: print(f"{nerr} optimisations have failed!\n")
    return nulls