import pandas as pd
import numpy as np
from scipy.optimize import minimize


class MultiClassLogisticRegression:
    def __init__(self, sigma=None, method="BFGS", disp=True):
        self.W = None
        self.sigma = sigma
        self.method = method
        self.disp = disp

    def fit(self, X, Y):
        N, D_w_bias = X.shape
        _, C = Y.shape
        initial_W_flat = np.zeros(D_w_bias * C)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        result = minimize(
            fun=self._cost,
            x0=initial_W_flat,
            args=(X, Y, self.sigma),
            method=self.method,
            jac=self._gradient,
            options={"disp": self.disp},
        )

        self.W = result.x.reshape(D_w_bias, C)
        return self.W

    def eval(self, X, Y, lr_only=False):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self._cost(self.W, X, Y, sigma=None, lr_only=lr_only)

    def _cost(self, W, X, Y, sigma, lr_only=False):
        """
        Compute the negative log-likelihood for multi-class
        logistic regression with L2 regularization (or MAP).

        params
        ------
        W : np.ndarray, shape (D + 1, C) or flattened (D+1 * C)
        weight matrix, will be in flattened form if in use
            for minimize() function
        X : pd.DataFrame, shape (N, D + 1)
            design matrix with bias column
        Y : np.ndarray, shape (N, C), where C = 3
            one-hot encoded choice labels for each trial as left,
            right or violation
        sigma : float (default=None)
            standard deviation of Gaussian prior, if None no
            regularization is applied
        lr_only : bool (default=False)
            whether to only consider the first two logits (L and R)
            when computing denominator of softmax

        returns
        -------
        - nll : float
            negative log-likelihood
        """
        if len(W.shape) == 1:
            W = W.reshape(X.shape[1], Y.shape[1])

        logits = X @ W

        # To compare binary & mutli-class models, only consider
        # the first two logits (L and R) in cost if lr is True
        if lr_only:
            log_sum_exp_term = self._log_sum_exp_lr(logits)
        else:
            log_sum_exp_term = self._log_sum_exp(logits)

        penalty = (
            (1 / (2 * (sigma**2))) * np.trace(W[1:, :].T @ W[1:, :]) if sigma else 0
        )
        nll = (
            -np.sum(Y * logits) + np.sum(self._log_sum_exp(log_sum_exp_term))
        ) + penalty
        return nll

    def _gradient(self, W, X, Y, sigma):
        """
        Compute the gradient of the negative log-likelihood for
        multi-class logistic regression with L2 regularization (or MAP).

        params
        ------
        W : np.ndarray, shape (D + 1, C) or flattened (D+1 * C)
        weight matrix, will be in flattened form if in use
        for minimize() function
        X : pd.DataFrame, shape (N, D + 1)
            design matrix with bias column
        Y : np.ndarray, shape (N, C), where C = 3
            one-hot encoded choice labels for each trial as left,
            right or violation
        sigma : float (default=None)
            standard deviation of Gaussian prior, if None no
            regularization is applied

        returns
        -------
        gradient :  np.ndarray, shape (D+1 * C)
            gradient of the negative log-likelihood

        """
        if len(W.shape) == 1:
            W = W.reshape(X.shape[1], Y.shape[1])

        logits = X @ W
        P = self._stable_softmax(logits)

        if sigma:
            penalty_gradient = W / (sigma**2)
        else:
            penalty_gradient = np.zeros_like(W)

        penalty_gradient[0, :] = 0  # No penalty for bias

        gradient = X.T @ (P - Y) + penalty_gradient
        return gradient.flatten()

    @staticmethod
    def _log_sum_exp(logits):
        max_logits = np.max(logits, axis=1, keepdims=True)
        return (
            np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True))
            + max_logits
        )

    @staticmethod
    def _log_sum_exp_lr(logits):
        max_logits = np.max(
            logits[:, :2], axis=1, keepdims=True
        )  # Only consider the first two logits (L and R)
        return (
            np.log(np.sum(np.exp(logits[:, :2] - max_logits), axis=1, keepdims=True))
            + max_logits
        )

    @staticmethod
    def _stable_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
        return exp_logits / sum_exp

    def generate_data(self, N, D, C, sigma=None, random_state=None):
        """
        Generate data from a multi-class logistic regression
        model with optional L2 regularization (or MAP).

        params
        ------
        N : int
            number of trials/samples
        D : int
            number of features
        C : int
            number of classes
        sigma : float (default=None)
            standard deviation of true weight matrix, if None
            generated with std of 1
        random_state : int (default=None)
            random seed

        returns
        -------
        w : np.ndarray, shape (D + 1, C)
            true weight matrix
        X : pd.DataFrame, shape (N, D + 1)
            design matrix with bias column
        Y : np.ndarray, shape (N, C)
            one-hot encoded choice labels for C classes
        """

        ## Design Matrix
        X = np.random.normal(size=(N, D))
        X = np.c_[np.ones(N), X]  # add bias column

        ## True Weights
        np.random.seed(random_state)
        if sigma:
            W = np.random.normal(loc=0, scale=sigma, size=(D + 1, C))
        else:
            W = np.random.normal(loc=0, scale=1, size=(D + 1, C))

        ## Choice Labels
        A = X @ W  # logits
        P = self._stable_softmax(A)
        Y = np.array([np.random.multinomial(1, n) for n in P])

        print(f"Generated {N} samples with {D} features and {C} classes")
        print(f"W is {W.shape} \nX is {X.shape} \nY is {Y.shape}")
        print(f"W has mean {np.mean(W):.3f} and std {np.std(W):.3f}")

        return W, X, Y
