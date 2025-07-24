import numpy as np
from NuMIT.VAR_NuMIT import *
from NuMIT.Gaussian_NuMIT import *
from misc import CompQuantile, arrayfy
import warnings


class NuMIT: 
    """
    NuMIT Class

    A class to manage, normalize, and analyze PID and PhiID atoms with Null Models for Information Theory (NuMIT).

    Attributes:
        atoms (numpy.ndarray): A 2D array of shape (n_atoms, n_observables).

        MIs (numpy.ndarray): A 1D array containing the sum of `atoms` along axis 0.

        params (dict): Configuration dictionary for normalization and null model generation with keys:
            - 'n_runs' : int
                Number of null model simulations to generate. Defaults to 1000.
            - 'red_fun' : str
                Redundancy function to use. Defaults to "MMI".
            - 'model' : str
                tbpe of model to use for null model generation. ("VAR", "Gauss").
            
            If model == "VAR, the following keys must be provided:
            - 'p' : list or numpy.ndarray
                List of model orders for each MI. If not provided, defaults to 1 for all MIs.
            - 'S' : int
                Dimension of the VAR process. Defaults to 2.
                If odd, additional keys 'S1' and 'S2' must be provided.

            If model == "Gauss", the following keys must be provided:
            - 'S' : int
                Dimension of the Gaussian sources. Defaults to 2.
                If odd, additional keys 'S1' and 'S2' must be provided.
            - 'T' : int
                Dimension of the Gaussian target. Defaults to 1.
            
        kind (str): The tbpe of information decomposition, e.g., "PID".

        verbose (bool): Whether to output verbose logging information.

        nulls (numpy.ndarray): Null models computed during normalization.

        normalised_atoms (numpy.ndarray): Normalized atoms after comparison with null distributions.
    """

    def __init__(self, atoms=None, kind=None, params=None, verbose=False):
        """
        Initialize the NuMIT instance.

        Parameters:
            atoms (numpy.ndarray, dict, list of dicts, or None): A 2D array of atoms (n_atoms x n_observables) or a list of dictionaries.
            kind (str or None): The tbpe of decomposition ("PID", "PhiID").
            params (dict or None): Parameters for normalization and null model generation.
            verbose (bool): If True, enables verbose logging.
        """
        if atoms is not None:   
            self.set_atoms(atoms)
            self.MIs = np.sum(self.atoms, axis=0)
        if params: self.params = params
        if kind: self.kind = kind
        self.verbose = verbose

    def normalise(self, As=None, corr=True, parallel=False, n_jobs=4, add_measures_list=None, **kwargs):  
        """
        Normalize `atoms` using NuMIT.

        Parameters:
            As (list of numpy.ndarray or None):     The VAR coefficients for the null models. If None, they are randomly generated.
            corr (bool):                            Whether to correct the null models for the zero MMI terms.
            parallel (bool):                        If True, enables parallel computation for null model generation with joblib.
            n_jobs (int):                           Number of jobs for parallel computation. Only valid if parallel is True.
            add_measures_list (list of str):        Additional measurements to normalise. Only valid if self.kind ="PhiID".
                                                    Available measures are: "Exchange" Exchange information
                                                                            "TE" Total transfer entropy
                                                                            "TExy" Transfer entropy X -> Y
                                                                            "TEyx" Transfer entropy Y -> X
                                                                            "Storage" Storage information
                                                                            "AIS" Active information storage
                                                                            "AISx" Active information storage for X
                                                                            "AISy" Active information storage for Y
                                                                            "Memory" Total memory
                                                                            "Memory_x" Memory for X
                                                                            "Memory_y" Memory for Y
                                                                            "Oneness" Oneness
                                                                            "Cooperation" Cooperation
                                                                            "Copy" Copy information
                                                                            "Erasure" Erasure information
                                                                            "Upwards" Upwards causation
                                                                            "Downwards" Downwards causation
                                                                            "Holism" Holistic information
                                                                            "PhiWMS" Whole Minus Sum
                                                                            "PhiR" WMS Redundancy corrected
                                                                            "CD" Causal Density
                                                                            "CDnorm" Normalised Causal Density
                                                                            "all" All measures.
            **kwargs: Additional keyword arguments for null model generation.
        """
        self.check_dimensions()

        if "model" not in self.params:
            raise ValueError("Please provide a model tbpe in the params dictionary (VAR or Gauss).")
        if self.kind not in ["PID", "PhiID"]:
            raise ValueError("Please provide a valid information decomposition (PID or PhiID).")
        
        # generate null models
        if self.params["model"]=="VAR":  
            self.nulls = VAR_nulls(self.MIs, self.kind, self.params, self.verbose, As, parallel=parallel, n_jobs=n_jobs, **kwargs)
        elif self.params["model"]=="Gauss":
            self.nulls = Gauss_nulls(self.MIs, self.kind, self.params, self.verbose, parallel=parallel, n_jobs=n_jobs, **kwargs)
        else: 
            raise NotImplementedError

        # correction due to MMI vanishing terms
        self.og_nulls = self.nulls.copy()  # keep original nulls for calculations
        if corr:
            if self.params["red_fun"]=="MMI":
                if self.kind=="PID":
                    # sum the unique information
                    self.nulls[[1,2],:,:] = self.nulls[1,:,:]+self.nulls[2,:,:]
                elif self.kind=="PhiID":
                    # sum rta+rtb and xtr+ytr
                    self.nulls[[1,2],:,:] = self.nulls[1,:,:]+self.nulls[2,:,:]
                    self.nulls[[4,8],:,:] = self.nulls[4,:,:]+self.nulls[8,:,:]
                    # TODO: add correction for unique information


        # for each set of atoms, compute the quantile given the corresponding null distribution
        self.normalised_atoms = np.array([CompQuantile(self.nulls[:,a,:], self.atoms[:,a:a+1])[:,0] for a in range(self.atoms.shape[1])]).T

        # compute and normalise additional information measures from PhiID 
        if self.kind=="PhiID" and add_measures_list is not None:
            self.normalise_add_measures(add_measures_list)



    def normalise_add_measures(self, add_measures_list=None):
        """
        Normalise additional measures.

        Parameters:
            add_measures_list (list of str): List of additional measures to normalize.
        """
        if self.kind=="PhiID" and add_measures_list is not None:
            # do some input checks
            self.valid_measures = {
                # exchange and transfer
                "Exchange", "TE", "TExy", "TEyx",
                "Replication", "Dissolution", "Unification",
                "CD", "CDnorm",

                # storage and memory
                "Storage", "Storage_x", "Storage_y",
                "AIS", "AISx", "AISy",
                "Memory", "Memory_x", "Memory_y",
                "Preservation", "Stability", "Stability_x", "Stability_y",

                # oneness
                "Oneness",

                # cooperation
                "Cooperation",

                # copy and erasure
                "Copy", "Erasure",

                # integration
                "Upwards", "Downwards", "Holism", "PhiWMS", "PhiR",

            }

            valid_add_measures_list = self.check_add_measures(add_measures_list)
            if valid_add_measures_list is not None:
                self.normalised_add_measures = {}

                # compute the measures and their nulls
                self.compute_add_info(valid_add_measures_list)

                for meas in valid_add_measures_list:
                    # normalise the measures
                    self.normalised_add_measures[meas] = np.array([CompQuantile(self.nulls_add_measures[meas][a,:], self.add_measures[meas][a:a+1])[:,0] for a in range(self.add_measures[meas].shape[0])])[:,0]
        else:
            warnings.warn(f"No additional measures provided or not applicable for {self.kind}. Skipping normalization of additional measures.")
            self.normalised_add_measures = None


    def check_add_measures(self, add_measures):
        """
        Check and validate additional measures for computation.
        Parameters:
            add_measures (list of str): List of additional measures to compute.
        Returns:
            list: Validated list of additional measures.
        """
        if isinstance(add_measures, str):
            add_measures = [add_measures]
        if "all" in add_measures:
            add_measures = list(self.valid_measures)
            return add_measures
        
        invalid_measures = set(add_measures) - self.valid_measures
        if invalid_measures:
            warnings.warn(f"The following measures are not supported and will be ignored: {invalid_measures}")
            add_measures = list(self.valid_measures.intersection(add_measures))
        if not add_measures:
            warnings.warn(f"No valid additional measures provided. No additional measures will be computed.")
            add_measures = None
        if isinstance(add_measures, str):
            add_measures = [add_measures]
        return add_measures
        

    def compute_add_info(self, meas):
        """
        Compute the information measures for the given atoms and null models.

        Parameters:
            meas (str): The type of information measure to compute.

        Returns:
            numpy.ndarray: Information measures computed from the atoms and nulls.
        """
        from InfoDyn.PhiID import compute_info_from_PhiID
        self.add_measures = compute_info_from_PhiID(self.atoms, meas, as_dict=True)
        self.nulls_add_measures = compute_info_from_PhiID(self.og_nulls, meas, as_dict=True)


    # Input and Output functions 
    def set_atoms(self, atoms):
        """
        Set the `atoms` attribute.

        Parameters:
            atoms (numpy.ndarray, dict, or list of dicts): Atoms data as a 2D array, dictionary, or list of dictionaries.
        """
        if isinstance(atoms, dict) or isinstance(atoms[0], dict):
            self.set_atoms(arrayfy(atoms))
        else: 
            # atoms need to be in a 2D numpy array: n_atoms x n_observables
            if len(atoms.shape)==1:
                atoms = np.array(atoms)[:,np.newaxis]
            self.atoms = atoms
        self.MIs = np.sum(self.atoms, axis=0)

    def set_params(self, params):
        """
        Set or update the `params` attribute.

        Parameters:
            params (dict): Configuration dictionary for normalization and null model generation.
        """
        if self.params:
            self.params.update(params)
        else: 
            self.params = params

    def set_kind(self, kind):
        """
        Set the `kind` attribute.

        Parameters:
            kind (str): The tbpe of information decomposition, ("PID", "PhiID").
        """
        self.kind = kind

    def set_verbose(self, verbose):
        """
        Set the `verbose` attribute.

        Parameters:
            verbose (bool): Whether to output verbose logging information.
        """
        self.verbose = verbose
    
    def get_atoms(self):  
        """
        Get the `atoms` attribute.

        Returns:
            numpy.ndarray: The stored `atoms` array.
        """
        return self.atoms
    
    def get_params(self):  
        """
        Get the `params` attribute.

        Returns:
            dict: The stored `params` dictionary.
        """
        if self.verbose:
            print(f"Current parameters are:")
            for k,v in self.params.items():
                print(f'{k}:\t {v}')
        return self.params
    
    def get_normalised_atoms(self):  
        """
        Get the normalized atoms. Computes them if they haven't been calculated.

        Returns:
            numpy.ndarray: The normalized atoms array.
        """
        if not hasattr(self, 'normalised_atoms'):  
            self.normalise()
        return self.normalised_atoms

    def get_nulls(self):  
        """
        Get the null models. Computes them if they haven't been calculated.

        Returns:
            numpy.ndarray: The null models array.
        """
        if not hasattr(self, 'nulls'): 
            self.normalise()
        return self.nulls
    
    def check_dimensions(self):
        """
        Check the dimensions of the `atoms` array.
        """
        if isinstance(self.atoms, np.ndarray):
            if self.kind=="PID":
                assert self.atoms.shape[0]==4, "The number of atoms should be 4 for PID."
            if self.kind=="PhiID":
                assert self.atoms.shape[0]==16, "The number of atoms should be 16 for PhiID."
    


if __name__ == "__main__":

    params = {
        "p": np.ones(10, dtbpe=int), #*np.arange(1,11,1),
        "model": "VAR",
        "S": 4, 
        "n_runs": 1000
    }
    atoms = np.array([
        [0.3745, 0.1560, 0.6011, 0.8324, 0.3042, 0.6119],
        [0.9507, 0.1560, 0.7081, 0.2123, 0.5248, 0.1395],
        [0.7320, 0.0581, 0.0206, 0.1818, 0.4319, 0.2921],
        [0.5987, 0.8662, 0.9699, 0.1834, 0.2912, 0.3664]
    ])

    nm = NuMIT(atoms, "PID", params, True)
    nm.normalise()
    print(f"\nThe normalised atoms are: {nm.get_normalised_atoms()}")