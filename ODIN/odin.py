from time import time
from .make_x import make_x
from .math_functions import *
from .mat_vec_transforms import half_vectorize_all


# Function to calculate threshold
def thresholder(y):  # max distance thresholder
    n_points = min([1000, len(y)])
    res = 1 / n_points
    p = np.arange(res, 1 + res, res)
    y_quantiles = np.quantile(y, p)
    slope = (y_quantiles[-1] - y_quantiles[0]) / (p[-1] - p[0])
    intercept = y_quantiles[0] - slope * p[0]
    distances = np.abs(y_quantiles - slope * p - intercept) / np.sqrt(1 + slope ** 2)
    return y_quantiles[np.argmax(distances)]


class ODIN:
    def __init__(self, lam=0.001, maxiter=10000, tol=1e-6):
        self.lam = lam
        self.tol = tol
        self.maxiter = maxiter
        self.x = None
        self.a = None
        self.z = None
        self.beta = None
        self.log_like = np.zeros(self.maxiter + 1)
        self.time = [0., 0.]
        self.im1 = None
        self.im2 = None
        self.thresh1 = None
        self.thresh2 = None
        self.outliers = None

    def __logistic_predict(self):
        return expit(self.z + np.matmul(self.x, self.beta))

    def __total_log_like(self):
        eta = self.z + np.matmul(self.x, self.beta)
        negative_log_like = - np.sum(self.a * eta - np.log(1 + np.exp(eta))) / self.beta.shape[1]
        return negative_log_like + (self.lam / 2) * np.sum(self.z ** 2)

    def __fit_model(self, print_out, init_z, init_beta):
        print("---Estimating Model Parameters---")
        start = time()

        l, n = self.a.shape  # Length of the half vectors and Sample Size
        p = self.x.shape[1]

        if init_z is None:
            self.z = np.zeros((l, 1))
        else:
            self.z = init_z

        if init_beta is None:
            self.beta = np.zeros((p, n))
        else:
            self.beta = init_beta

        # Iteration parameters
        diff = np.inf
        iterate = 0

        # Keep track of the likelihood
        # norms = np.zeros(self.maxiter)

        # Total log likelihood at initial values
        self.log_like[iterate] = self.__total_log_like()

        xtx = np.linalg.inv(np.matmul(np.transpose(self.x), self.x))
        q = np.eye(l) - np.matmul(np.matmul(self.x, xtx), np.transpose(self.x))
        r = 4 * np.linalg.inv(4 * self.lam * np.eye(l) + q)

        while np.absolute(diff) > self.tol and iterate < self.maxiter:
            iterate = iterate + 1

            # Predicted value at current estimates
            b = self.__logistic_predict()

            # current residue
            res = self.a - b

            # MM
            delta_z = np.matmul(r, -self.lam * self.z + np.matmul(q, np.mean(res, 1).reshape(l, 1)))
            delta_beta = 4 * np.matmul(xtx, np.matmul(np.transpose(self.x), res - n * delta_z))
            self.z = self.z + delta_z
            self.beta = self.beta + delta_beta

            # New total log likelihood
            self.log_like[iterate] = self.__total_log_like()

            # Negative of Gradient
            # direction_z = -self.lam * self.z + np.mean(res, 1).reshape(l, 1)
            # direction_beta = np.matmul(np.transpose(self.x), res) / n
            # Norm of gradient
            # norms[iterate - 1] = l2_norm(np.array([l2_norm(direction_z), l2_norm(direction_beta)]))

            diff = (self.log_like[iterate - 1] - self.log_like[iterate]) / self.log_like[iterate - 1]

            if print_out:
                print(f"{iterate}: Total likelihood: {self.log_like[iterate]}; Relative likelihood change: {diff}")

        self.log_like = self.log_like[0:(iterate + 1)]
        self.time[0] = float(time() - start)
        print(f"Estimated in {iterate} iteratations in {round(self.time[0], 3)} seconds.")
        return

    def __calculate_influence(self, print_out):
        print("---Calculating Influence Measures---")
        start = time()
        l = self.z.shape[0]
        pp, n = self.beta.shape

        beta_bar = np.mean(self.beta, 1).reshape(self.beta.shape[0], 1)
        s = np.matmul(self.beta, np.transpose(self.beta)) / n - np.matmul(beta_bar, np.transpose(beta_bar))
        s_inv = np.linalg.inv(s)
        beta_diff = self.beta - beta_bar

        z_influence = np.zeros((l, n))
        self.im1 = np.zeros(n)
        self.im2 = np.zeros(n)

        b = self.__logistic_predict()

        t = np.diag(np.sum(b * (1 - b), 1) + self.lam * n)

        iter_start = time()
        for i in range(n):

            if i % 100 == 0 and print_out:
                print(f"At subject {i}")

            p = b[:, i]
            w = p * (1 - p)
            wx = w[:, np.newaxis] * self.x
            xwx_inv = np.linalg.inv(np.matmul(np.transpose(self.x), wx))
            t -= np.matmul(np.matmul(wx, xwx_inv), np.transpose(wx))
            z_influence[:, i] = -self.lam * self.z.reshape(l) + (self.a[:, i] - p)

            beta_i_diff = beta_diff[:, i].reshape(pp, 1)
            self.im2[i] = quadratic_form(beta_i_diff, s_inv)

        t = np.linalg.inv(t)
        z_influence = (n / (n - 1)) * np.matmul(t, z_influence)

        for i in range(n):
            self.im1[i] = l2_norm(z_influence[:, i])

        print(f"Calculated influence in {round(time() - start, 3)} seconds.")
        self.time[1] = float(time() - iter_start)
        return

    def __calculate_thresholds(self):
        print("---Thresholding Influence Measures---")
        self.thresh1 = thresholder(self.im1)
        self.thresh2 = thresholder(self.im2)
        out1 = np.where(self.im1 > self.thresh1)
        out2 = np.where(self.im2 > self.thresh2)
        self.outliers = np.sort(np.union1d(out1, out2))
        return

    def fit_and_detect_outliers(self, a, lobes=None, hemis=None, x=None, init_z=None, init_beta=None, print_out=False):
        start = time()
        if len(a.shape) == 3:
            self.a = half_vectorize_all(a)
        else:
            self.a = a
        if x is None:
            self.x = make_x(lobes, hemis)
        else:
            self.x = x
        self.__fit_model(print_out, init_z, init_beta)
        self.__calculate_influence(print_out)
        self.__calculate_thresholds()
        print(f"ODIN finished in {round(time() - start, 3)} seconds")
        return
