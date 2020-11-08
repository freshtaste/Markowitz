import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold


class Markowitz(object):
    
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
        self.mu, self.cov = self.est_mean(), self.est_cov()
        self.w = None
    
    
    def vanilla(self, sigma):
        theta = self.mu.dot(np.linalg.inv(self.cov).dot(self.mu))
        w = sigma/np.sqrt(theta)* np.linalg.inv(self.cov).dot(self.mu)
        self.w = w/np.sum(w)
        return self.w
    
    
    def maxser(self, sigma, kfolds=10):
        """
        Fit the data to find the optimized portfolio weights in Scenario I

        Parameters
        ----------
        sigma : float
            The given risk level.
        kfolds : int
            Number of folds for cross validation.

        Returns
        -------
        w : np.array
            The optimal weights.

        """
        # 1. Estimate the square of the maximum Sharpe ratio and compute response
        theta = self.est_sharp_ratio()
        if theta < 0:
            raise RuntimeError("theta is negative.")
        rc = sigma*(1+theta)/np.sqrt(theta)
        Rc = np.ones(self.T)*rc
        # 2. select lambda by cross validation
        kf = KFold(n_splits=kfolds, shuffle=False)
        alphas = list()
        for train_index, valid_index in kf.split(self.R.T):
            alphas.append(Markowitz.est_and_valid(self.R.T[train_index], 
                Rc[train_index], self.R.T[valid_index], sigma))
        myAlpha = np.mean(alphas)
        # 3. solve optimal weights with the average best alpha
        lasso = linear_model.Lasso(alpha=myAlpha, fit_intercept=False)
        w = lasso.fit(self.R.T, Rc).coef_
        self.w = w/np.sum(w)
        return self.w
        
        
    @staticmethod
    def est_and_valid(X_train, y_train, X_valid, sigma):
        """
        Use validation set to select the best alpha among all lasso paths

        Parameters
        ----------
        X_train : np.array
            training set feature.
        y_train : np.array
            training set response.
        X_valid : np.array
            valid set feature.
        sigma : float
            target risk level.

        Returns
        -------
        alpha_opt : float
            best alpha.

        """
        alphas, _, coefs = linear_model.lars_path(X_train, y_train, method='lasso')
        variances = np.var(X_valid.dot(coefs),axis=0)
        idx_opt = np.argmin(np.abs(np.sqrt(variances)-sigma))
        alpha_opt = alphas[idx_opt]
        return alpha_opt
    
        
    def est_mean(self):
        return np.mean(self.R,axis=1)
    
        
    def est_cov(self):
        """
        Covariance estimation. Could be replaced with better method later.

        """
        return np.cov(self.R)
    
    
    def est_sharp_ratio(self):
        theta_s = self.mu.dot(np.linalg.inv(self.cov).dot(self.mu))
        theta = ((self.T - self.N - 2)*theta_s - self.N)/self.T
        return theta
    
    
    def insample(self):
        rp = self.w.dot(self.R)
        mu, sigma = np.mean(rp), np.sqrt(np.var(rp))
        return mu, sigma