import numpy as np


class COV(object):
    """ large covriance estimation"""
    
    def __init__(self, R):
        """
        Parameters
        ----------
        R : Numpy Array
            Return matrix N by T.
            
        """
        self.R = R
        self.N = R.shape[0]
        self.T = R.shape[1]
        self.mu, self.cov = np.mean(R, axis=1), np.cov(R)
        
    
    def poet_known(self, K, tau):
        """
        Estimate covariance matrix with POET algorithm

        Parameters
        ----------
        K : int
            Number of factors.
        tau : float
            Thresholding for sparcity of residual matrices. Between 0 and 1

        Returns
        -------
        result : numpy.ndarray
            Estimated covariance matrix.

        """
        # check K and tau
        if K < 0 or K >= self.N:
            raise RuntimeError("Invalid value for K, number of factors.")
        if tau < 0 or tau > 1:
            raise RuntimeError("Invalid value for tau, which should be between 0 and 1.")
        
        # mean centering and calculate SVD for pca
        Rc = self.R.T - np.mean(self.R, axis=1)
        u, s, vt = np.linalg.svd(Rc/np.sqrt(self.T-1))
        eigvecs = vt.T
        eigvals = s**2
        
        # decomposition of covariance matrix
        cov_pca = eigvecs[:,:K] @ np.diag(eigvals[:K]) @ eigvecs[:,:K].T
        Rk = self.cov - cov_pca
        
        # thresholding the complement matrix
        rii = np.diag(Rk)
        tauij = np.sqrt(np.outer(rii, rii))*tau
        RkT = Rk*(Rk > tauij)
        
        # combine the two terms
        result = cov_pca + RkT
        return result