import numpy as np
import os
from VAR.VAR_fitness import fit_var
from VAR.VAR_utils import var_to_covariance
from misc import get_gaussian_cov, saveData
from InfoDyn import measures


class Info_calculator:
    def __init__(
        self,
        data,
        model="Gauss",
        p=None,
        maxp=1,
        detrend=True,
        saveA=False,
        saveV=False,
        pathA="Results/A.pickle",
        pathV="Results/V.pickle",
        verbose=True,
    ):
        """
        Parameters:
            data: np.ndarray or np.ndarray (2D/3D timeseries data or covariance matrix)
            model: str, "Gauss" or "VAR"
            p: int, number of past lags (only for VAR model)
            maxp: int, maximum number of past lags to consider (only for VAR model)
            detrend: bool, whether to detrend the data
        """
        if data.shape[0] < data.shape[1]:
            # data is a time series
            if model not in ["Gauss", "VAR"]:
                raise ValueError("Model must be 'Gauss' or 'VAR'.")
            self.nvar = data.shape[0]
            if model == "Gauss":
                if data.shape[0] < 2:
                    raise ValueError("Time series must be at least bivariate.")
                self.cov = get_gaussian_cov(data, lag=1, detrend=detrend)
                self.p = 1

            elif model == "VAR":
                A, V, p, _ = fit_var(data, maxp=maxp, detrend=detrend)
                self.cov = var_to_covariance(A, V, p)
                self.A, self.V, self.p = A, V, p
                # save the VAR parameters if requested
                if saveA:
                    if not os.path.exists(pathA.split("/")[0]):
                        os.makedirs(pathA.split("/")[0])
                    saveData(A, pathA)
                if saveV:
                    if not os.path.exists(pathV.split("/")[0]):
                        os.makedirs(pathV.split("/")[0])
                    saveData(V, pathV)

        elif data.ndim == 2 and data.shape[0] == data.shape[1]:
            # data is a covariance matrix
            self.cov = data
            self.p = 1 if p is None else p
            assert (
                data.shape[0] / (self.p + 1)
            ).is_integer(), "Error in the input of the covariance or past lags. Covariance matrix must have a shape of (n*(p+1), n*(p+1))."
            self.nvar = data.shape[0] // (self.p + 1)
        else:
            raise ValueError("Input data must be a 2D/3D array or a covariance matrix.")

        if verbose:
            print(
                f"Initialising Info_calculator. \nThe system has {self.nvar} variables and {self.p} past lag(s). "
                # f"Covariance of the system is of shape {self.cov.shape}."
            )

    def get_source_index(self, idx):
        """
        Get the source index for a given source in the covariance matrix.
        If p>1, it returns all the time lagged indeces for that source.
        """
        idx = np.asarray(idx)
        assert np.all(idx < self.nvar), "Index out of bounds for the covariance matrix."

        # print("sources: ", (idx + self.nvar * np.arange(self.p)).tolist())
        return np.add.outer(idx, self.nvar * np.arange(self.p)).ravel().tolist()
        # return (idx + self.nvar * np.arange(self.p)).tolist()

    def get_target_index(self, idx):
        """
        Get the target index for a given target in the covariance matrix.
        """
        idx = np.asarray(idx)
        assert np.all(idx < self.nvar), "Index out of bounds for the covariance matrix."

        # print("targets: ", (idx + self.nvar * np.array([self.p])).tolist())
        return (idx + self.nvar * np.array([self.p])).tolist()

    def istantaneous_mutual_information(
        self, idx1, idx2, time="past", pert=False, **kwargs
    ):
        """
        Compute the instantaneous mutual information between two variables.
        Parameters:
            idx1: int or list of ints, index of the first variable(s).
            idx2: int or list of ints, index of the second variable(s).
            time: str, "past" or "future", to specify the moment in time.
        Returns:
            float: Instantaneous mutual information in bits.
        """
        if time == "past":
            idx1 = self.get_source_index(idx1)
            idx2 = self.get_source_index(idx2)
        elif time == "future":
            idx1 = self.get_target_index(idx1)
            idx2 = self.get_target_index(idx2)

        if not pert:
            return measures.mutual_information(self.cov, idx1, idx2)
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.mutual_information(self.pert_cov, idx1, idx2)

    def time_delayed_mutual_information(self, idx1, idx2, pert=False, **kwargs):
        """
        Compute the time delayed mutual information between two variables.
        Parameters:
            idx1: int or list of ints, index of the past variable(s).
            idx2: int or list of ints, index of the future variable(s).
        Returns:
            float: Time delayed mutual information in bits.
        """
        # past and future indices
        idx1 = self.get_source_index(idx1)
        idx2 = self.get_target_index(idx2)

        if not pert:
            return measures.mutual_information(self.cov, idx1, idx2)
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.mutual_information(self.pert_cov, idx1, idx2)

    def transfer_entropy(self, source, target, pert=False, **kwargs):
        """
        Compute the transfer entropy from one variable to another.
        Parameters:
            source: int or list of ints, index of the source variable(s).
            target: int or list of ints, index of the target variable(s).
        Returns:
            float: Transfer entropy in bits.
        """
        idx_Spast = self.get_source_index(source)
        idx_Tpast = self.get_source_index(target)
        idx_Tfuture = self.get_target_index(target)

        if not pert:
            return measures.transfer_entropy(
                self.cov, idx_Tfuture, idx_Tpast, idx_Spast
            )
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.transfer_entropy(
                self.pert_cov, idx_Tfuture, idx_Tpast, idx_Spast
            )

    def pearson_correlation(self, idx1, idx2, time="past", pert=False, **kwargs):
        """
        Compute the Pearson correlation between two variables.
        Parameters:
            idx1: int or list of ints, index of the first variable(s).
            idx2: int or list of ints, index of the second variable(s).
            time: str, "past" or "future", to specify the moment in time.
        Returns:
            float: Pearson correlation.
        """
        if time == "past":
            # get only the last time lag (in case p>1)
            idx1 = self.get_source_index(idx1)[0]
            idx2 = self.get_source_index(idx2)[0]
        elif time == "future":
            idx1 = self.get_target_index(idx1)
            idx2 = self.get_target_index(idx2)

        if not pert:
            return measures.pearson_correlation(self.cov, idx1, idx2)
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.pearson_correlation(self.pert_cov, idx1, idx2)

    def PID(self, idx_source1, idx_source2, idx_target, pert=False, **kwargs):
        """
        Compute the Partial Information Decomposition (PID) between two sets of variables.
        Parameters:
            idx_source1: list of ints, indices of the first source variables.
            idx_source2: list of ints, indices of the second source variables.
            idx_target: list of ints, indices of the target variable(s).
            **kwargs: Additional keyword arguments for PID calculation.
        Returns:
            dict or np.array: PID components [Redundancy, Unique_X, Unique_Y, Synergy].
        """
        s1 = self.get_source_index(idx_source1)
        s2 = self.get_source_index(idx_source2)
        t = self.get_target_index(idx_target)

        if not pert:
            return measures.PID(self.cov, s1, s2, t, **kwargs)
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.PID(self.pert_cov, s1, s2, t, **kwargs)

    def PhiID(self, idx_var1, idx_var2, pert=False, **kwargs):
        """
        Compute the PhiID between two sets of variables.
        Parameters:
            idx_var1: list of ints, indices of the first variable(s).
            idx_var2: list of ints, indices of the second variable(s).
            **kwargs: Additional keyword arguments for PhiID calculation.
        Returns:
            dict or np.array: PhiID components.
        """
        assert self.p == 1, "PhiID is only implemented for p=1."
        s1 = self.get_source_index(idx_var1)
        s2 = self.get_source_index(idx_var2)
        t1 = self.get_target_index(idx_var1)
        t2 = self.get_target_index(idx_var2)

        if not pert:
            return measures.PhiID(self.cov, s1, s2, t1, t2, **kwargs)
        else:
            if not hasattr(self, "pert_cov"):
                self.perturb_covariance(**kwargs)
            return measures.PhiID(self.pert_cov, s1, s2, t1, t2, **kwargs)

    def perturb_covariance(self, cut="full", ps=None, **kwargs):
        """
        Apply causal perturbation to the covariance matrix
        by applying a maximum entropy distribution

        Parameters:
            cut: str, type of cut ("full" or "partial")
            ps: list of list of ints, source partitions for the partial cut (only for "partial" cut)
                e.g. ps = [[0, 1], [2]] for 3 variables where sources 0 and 1 are in one partition
        Returns:
            None: updates the perturbed covariance matrix as self.pert_cov
        """
        # dimension of the source variables (possibly with past lags)
        dimsrc = self.nvar * self.p

        # partition the covariance matrix into past and future
        Spp = self.cov[:dimsrc, :dimsrc]
        Spp_inv = np.linalg.inv(Spp)
        Spf = self.cov[:dimsrc, dimsrc:]
        Sff = self.cov[dimsrc:, dimsrc:]

        # compute conditional and transition matrix
        A = Spf.T @ Spp_inv
        Sf_p = Sff - Spf.T @ Spp_inv @ Spf

        # introduce maxent distribution with fixed marginals
        if cut == "full":
            # destroy all correlations
            maxent_cov = np.diag(np.diag(Spp))
        elif cut == "partial":
            # destroy correlations between partitions, but not within them
            if ps is None:
                ps = list(np.arange(self.nvar))
            else:
                tot_len = 0
                for p in ps:
                    tot_len += len(p)
                if tot_len < self.nvar:
                    raise ValueError("For partial cut, all sources must be specified.")
                elif tot_len > self.nvar:
                    raise ValueError(
                        "For partial cut, too many sources were specified."
                    )

            maxent_cov = np.zeros((dimsrc, dimsrc))
            for p in ps:
                indx = self.get_source_index(p)
                maxent_cov[np.ix_(indx, indx)] = Spp[np.ix_(indx, indx)]

        # build perturbed covariance
        Q = np.zeros_like(self.cov)
        Q[:dimsrc, :dimsrc] = maxent_cov
        Q[:dimsrc, dimsrc:] = maxent_cov @ A.T
        Q[dimsrc:, :dimsrc] = Q[:dimsrc, dimsrc:].T
        Q[dimsrc:, dimsrc:] = Sf_p + A @ maxent_cov @ A.T

        self.pert_cov = Q
