"""
 FUNCTIONS FOR FITTING LOGISTIC
"""
import numpy as np
from time import time

# MATHEMATICAL FUNCTIONS


def expit(x):
    return np.where(x < 500, np.exp(x) / (1 + np.exp(x)), 1.0)


def l2_norm(x):
    return np.sqrt(np.sum(x ** 2))


def logit(x):
    return np.log(x / (1 - x))


def quadratic_form(x, a):
    return (np.dot(x.T, np.dot(a, x)))[0, 0]


# PREDICTION AND LOG-LIKELIHOOD FUNCTION


def logistic_predict(z, beta, x):
    return expit(z + np.matmul(x, beta))


def negative_log_like(z, beta, x, a):
    eta = z+np.matmul(x, beta)
    return - np.sum(a * eta - np.log(1 + np.exp(eta))) / beta.shape[1]


def total_log_like(z, beta, x, a, lam):
    return negative_log_like(z, beta, x, a) + (lam / 2)*np.sum(z ** 2)


# THE MAIN MM ALGORITHM
#
# This is the main function which fits the model using
# Majorization - Maximization (MM) algorithm.
#
# Input:
# a = a numpy matrix containing as it's columns the binary adjacency
#     matrices after vectorizing the lower triangular part.
# x = a numpy matrix containing the design matrix corresponding to
#     one individual
# lam = the value of lambda
# tol = the tolerance value for the stopping condition
# max_iterate = the maximum number of iterations
# init_z = the user provided initial value of z (if any)
# init_beta = the user provided initial value of beta (if any)
# print_out = boolean to determine if to print current likelihood and
#             other information after each iteration
#
# Output:
# A list containing:
# - the fitted value of z
# - the fitted value of beta
# - the sequence of total negative log-likelihood values over the iterations
#
def mm_logistic(a, x, lam=0.001, tol=1e-7, max_iterate=10000, init_z=None, init_beta=None, print_out=False):
    start = time()

    n = a.shape[1]  # Sample Size
    l = a.shape[0]  # Length of the half vectors
    p = x.shape[1]

    if init_z is None:
        z = np.zeros((l, 1))
    else:
        z = init_z
    if init_beta is None:
        beta = np.zeros((p, n))
    else:
        beta = init_beta

    # Iteration parameters
    diff = np.inf
    iterate = 0

    # Keep track of the likelihood
    log_like = np.zeros(max_iterate + 1)
    norms = np.zeros(max_iterate)

    # Total log likelihood at initial values
    log_like[iterate] = total_log_like(z, beta, x, a, lam)

    xtx = np.linalg.inv(np.matmul(np.transpose(x), x))
    q = np.eye(l) - np.matmul(np.matmul(x, xtx), np.transpose(x))
    r = 4 * np.linalg.inv(4 * lam * np.eye(l) + q)

    while np.absolute(diff) > tol and iterate < max_iterate:
        iterate = iterate + 1

        # Predicted value at current estimates
        b = logistic_predict(z, beta, x)
        # current residue
        res = a - b

        # MM
        delta_z = np.matmul(r, -lam * z + np.matmul(q, np.mean(res, 1).reshape(l, 1)))
        delta_beta = 4 * np.matmul(xtx, np.matmul(np.transpose(x), res - n * delta_z))
        z = z + delta_z
        beta = beta + delta_beta

        # New total log likelihood
        log_like[iterate] = total_log_like(z, beta, x, a, lam)

        # Negative of Gradient
        direction_z = -lam * z + np.mean(res, 1).reshape(l, 1)
        direction_beta = np.matmul(np.transpose(x), res) / n

        # Norm of gradient
        norms[iterate - 1] = l2_norm(np.array([l2_norm(direction_z), l2_norm(direction_beta)]))
        diff = (log_like[iterate - 1] - log_like[iterate]) / log_like[iterate - 1]

        if print_out:
            print(f"{iterate}: Total likelihood: {log_like[iterate]}; Gradient norm: {norms[iterate - 1]}; "
                  f"Relative likelihood change: {diff}")

    log_like = log_like[0:(iterate + 1)]
    t = time() - start
    print(f"While loop ended with {iterate} iterates in {t} seconds.")
    return [z, beta, log_like, t]


# CALCULATION OF INFLUENCE
#
# This is the function that calculates the influence measures
#
# Input:
# list1 = the output from mm_logistic.
# a = a numpy matrix containing as it's columns the binary adjacency
#     matrices after vectorizing the lower triangular part.
# x = a numpy matrix containing the design matrix corresponding to
#     one individual
# lam = the value of lambda
#
# Output:
# A list containing:
# - the influence vector whose norm is IM(i) stores as an array (l x n)
# - IM(i) for all i as an n-vector
# - d(beta_i) for all i as an n-vector
#
def influence_measure(list1, a, x, lam, print_out=True):
    start_time = time()
    z = list1[0]
    beta = list1[1]

    l = z.shape[0]
    n = beta.shape[1]

    beta_bar = np.mean(beta, 1).reshape(beta.shape[0], 1)
    s = np.matmul(beta, np.transpose(beta)) / n - np.matmul(beta_bar, np.transpose(beta_bar))
    s_inv = np.linalg.inv(s)
    beta_diff = beta - beta_bar

    z_influence = np.zeros((l, n))
    z_influence_1d = np.zeros(n)
    d_beta = np.zeros(n)

    b = logistic_predict(z, beta, x)

    t = np.diag(np.sum(b * (1 - b), 1) + lam * n)

    iter_start_time = time()
    for i in range(n):
        if i % 100 == 0 and print_out:
            print(i)
        p = b[:, i]
        w = p * (1 - p)
        wx = w[:, np.newaxis] * x
        xwx_inv = np.linalg.inv(np.matmul(np.transpose(x), wx))
        t -= np.matmul(np.matmul(wx, xwx_inv), np.transpose(wx))
        z_influence[:, i] = -lam * z.reshape(l) + (a[:, i] - p)

        beta_i_diff = beta_diff[:, i].reshape(beta.shape[0], 1)
        d_beta[i] = quadratic_form(beta_i_diff, s_inv)

    t = np.linalg.inv(t)
    z_influence = (n / (n - 1)) * np.matmul(t, z_influence)

    for i in range(n):
        z_influence_1d[i] = l2_norm(z_influence[:, i])

    print(f"Calculated influence in {time()-start_time} seconds.")
    return [z_influence, z_influence_1d, d_beta, time()-iter_start_time]
