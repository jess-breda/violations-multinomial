import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize


def fit_multiclass(X, Y, sigma=None, disp=True):
    """
    function to fit multiclass logistic regression model
    using scipy.optimize.minimize() function

    params
    ------
    X : pd.DataFrame, shape (N, D + 1)
        design matrix with bias column
    Y : np.ndarray, shape (N, C), where C = 3
        one-hot encoded choice labels for each trial as left,
        right or violation
    sigma : float (default=None)
        standard deviation of Gaussian prior, if None no
        regularization is applied
    disp : bool (default=True)
        if True, print convergence message from minimize()

    returns
    -------
    W : np.ndarray, shape (D + 1, C)
        optimized weights matrix
    """
    # initialize dimensions
    N = len(X)
    D = X.shape[1] - 1
    C = Y.shape[1]

    # initalize weights
    initial_W_flat = np.zeros((D + 1) * C)

    # Optimization using BFGS
    result = minimize(
        fun=multiclass_logistic_cost,
        x0=initial_W_flat,
        args=(X.to_numpy(), Y, sigma),
        method="BFGS",
        jac=multiclass_logistic_gradient,
        options={"disp": disp},
    )

    return result.x.reshape(D + 1, C)


## Function for Multi-class Logistic Regression
def log_sum_exp(logits):
    """
    Compute the log of the sum of exponential sin a
    numerically stable way by subtracting off the largest
    logit.
    """
    max_logits = np.max(logits, axis=1, keepdims=True)
    return (
        np.log(np.sum(np.exp(logits - max_logits), axis=1, keepdims=True)) + max_logits
    )


def stable_softmax(logits):
    # use log-sum-exp for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))

    # sum over classes & normalize
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
    return exp_logits / sum_exp


def multiclass_logistic_cost(W, X, Y, sigma=None):
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

    returns
    -------
    - nll : float
        negative log-likelihood
    """

    # reshape if from flat -> matrix if needed
    if len(W.shape) == 1:
        _, D_w_bias = X.shape
        _, C = Y.shape
        W = W.reshape(D_w_bias, C)

    logits = X @ W

    if sigma:
        penalty = 1 / (2 * (sigma**2)) * np.trace(W[1:, :].T @ W[1:, :])
    else:
        penalty = 0

    nll = (-np.sum(Y * logits) + np.sum(log_sum_exp(logits))) + penalty
    return nll


def multiclass_logistic_gradient(W, X, Y, sigma=None):
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

    # reshape if from flat -> matrix if needed
    if len(W.shape) == 1:
        _, D_w_bias = X.shape
        _, C = Y.shape
        W = W.reshape(D_w_bias, C)

    logits = X @ W
    P = stable_softmax(logits)

    if sigma:
        penalty_gradient = W / (sigma**2)
        penalty_gradient[0, :] = 0  # No penalty for bias
    else:
        penalty_gradient = 0

    gradient = X.T @ (P - Y) + penalty_gradient
    return gradient.flatten()


def session_train_test_split(df, test_size, random_state=55):
    """
    Function for doing a train/test split on a dataframe with
    multiple sessions. This function will split the data by session
    and return a train and test dataframe.

    params
    ------
    df : pd.DataFrame
        dataframe with multiple sessions to split for a single
        animal, likely created by get_rat_viol_data()
    test_size : float
        proportion of data to use for test set
    random_state : int (default: 55)
        random state for reproducibility

    returns
    -------
    train_df : pd.DataFrame
        dataframe with data from sessions in train set
    test_df : pd.DataFrame
        dataframe with data from sessions in test set
    """
    unique_values = df["session"].unique()
    train_values, test_values = train_test_split(
        unique_values, test_size=test_size, random_state=random_state
    )

    train_df = df[df["session"].isin(train_values)]
    test_df = df[df["session"].isin(test_values)]

    return train_df, test_df


def generate_model_names(params):
    tausL = params["tausL"]
    names = []

    for tau_list in tausL:
        if tau_list[0] is None:
            names.append("base")
        else:
            name = "t_" + "_".join(map(str, tau_list))
            names.append(name)

    params["names"] = names
    return params
