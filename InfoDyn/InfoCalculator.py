import numpy as np
import os 
from VAR.VAR_fitness import fit_var
from VAR.VAR_utils import var_to_covariance
from misc import get_gaussian_cov, saveData
from InfoDyn import measures

class Info_calculator:
    def __init__(self, data, model="Gauss", p=None, maxp=1, detrend=True, saveA=False, pathA="Results/A.pickle"):
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
            self.dim = data.shape[0]
            if model == "Gauss":
                if data.shape[0] < 2:
                    raise ValueError("Time series must be at least bivariate.")
                self.cov = get_gaussian_cov(data, lag=1, detrend=detrend)
                self.p = 1

            elif model == "VAR":
                A, V, p, _ = fit_var(data, maxp=maxp, detrend=detrend)
                self.cov = var_to_covariance(A, V, p)
                self.A, self.V, self.p = A, V, p
                # save the VAR coefficients if requested
                if saveA:
                    if not os.path.exists(pathA.split("/")[0]):
                        os.makedirs(pathA.split("/")[0])
                    saveData(A, pathA)

        elif data.ndim==2 and data.shape[0] == data.shape[1]:
            # data is a covariance matrix
            assert data.shape[0]%2==0, "Covariance matrix of a dynamical system must have even dimensions."
            self.cov = data
            if p is None:
                self.dim = data.shape[0] // 2
                self.p = 1
        else:
            raise ValueError("Input data must be a 2D/3D array or a covariance matrix.")
        

    def get_source_index(self, idx):
        """
        Get the source index for a given source in the covariance matrix.
        If p>1, it returns all the time lagged indeces for that source.
        """
        idx = np.asarray(idx)
        assert np.all(idx < self.dim), "Index out of bounds for the covariance matrix."

        # print("sources: ", (idx + self.dim * np.arange(self.p)).tolist())
        return (idx + self.dim * np.arange(self.p)).tolist()
    

    def get_target_index(self, idx):
        """
        Get the target index for a given target in the covariance matrix.
        """
        idx = np.asarray(idx)
        assert np.all(idx < self.dim), "Index out of bounds for the covariance matrix."

        # print("targets: ", (idx + self.dim * np.array([self.p])).tolist())
        return (idx + self.dim * np.array([self.p])).tolist()


    def istantaneous_mutual_information(self, idx1, idx2, time="past"):
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

        return measures.mutual_information(self.cov, idx1, idx2)
    

    def time_delayed_mutual_information(self, idx1, idx2):
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

        return measures.mutual_information(self.cov, idx1, idx2)


    def transfer_entropy(self, source, target):
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

        return measures.transfer_entropy(self.cov, idx_Tfuture, idx_Tpast, idx_Spast)


    def pearson_correlation(self, idx1, idx2, time="past"):
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

        return measures.pearson_correlation(self.cov, idx1, idx2)


    def PID(self, idx_source1, idx_source2, idx_target, **kwargs):
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
        return measures.PID(self.cov, s1, s2, t, **kwargs)
    

    def PhiID(self, idx_var1, idx_var2, **kwargs):
        """
        Compute the PhiID between two sets of variables.
        Parameters:
            idx_var1: list of ints, indices of the first variable(s).
            idx_var2: list of ints, indices of the second variable(s).
            **kwargs: Additional keyword arguments for PhiID calculation.
        Returns:
            dict or np.array: PhiID components.
        """
        s1 = self.get_source_index(idx_var1)
        s2 = self.get_source_index(idx_var2)
        t1 = self.get_target_index(idx_var1)
        t2 = self.get_target_index(idx_var2)
        return measures.PhiID(self.cov, s1, s2, t1, t2, **kwargs)