import numpy as np


def expit(x):
    return np.where(x < 500, np.exp(x) / (1 + np.exp(x)), 1.0)


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def logit(x):
    return np.log(x / (1 - x))


def quadratic_form(x, a):
    return (np.dot(x.T, np.dot(a, x)))[0, 0]
