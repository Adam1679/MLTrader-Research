import numpy as np


## function to calculate t-stat
def get_t_value(train_mat, signal, response):
    beta = np.sum(train_mat[signal] * train_mat[response]) / sum(
        train_mat[signal] ** 2
    )  ## regressio coef
    sigma = np.sqrt(
        np.sum((train_mat[signal] * beta - train_mat[response]) ** 2)
        / (len(train_mat) - 1)
    )
    v = np.sqrt(
        np.sum(train_mat[signal] ** 2)
    )  ## sigma/v is the standard devication of beta_hat
    return beta / sigma * v
