import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, Sum, Product
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

domain = np.array([[0, 5]])

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # speed threshold
        self.kappa = 1.2 #0.21

        # Validation Function F
        self.f_noise = 0.15  # 0.5?
        self.f_var = 0.5
        self.f_lenscale = 0.5
        self.f_nu = 2.5

        k_1 = Matern(length_scale=self.f_lenscale, length_scale_bounds="fixed", nu=self.f_nu)
        k_2 = ConstantKernel(constant_value=self.f_var, constant_value_bounds="fixed")
        k_3 = WhiteKernel(noise_level=self.f_noise)
        self.f_kernel = Sum(Product(k_1, k_2), k_3)
        # Default GaussianProcessRegressor (check argument possibilities)
        self.gp_f = GaussianProcessRegressor(kernel=self.f_kernel, alpha=1e-10,
                                             optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False,
                                             copy_X_train=True, random_state=None)

        # Speed Function V
        self.v_noise = 0.0001
        self.v_var = np.sqrt(2)
        self.v_lenscale = 0.5
        self.v_mean = 1.5
        self.v_nu = 2.5

        kv_1 = Matern(length_scale=self.v_lenscale, length_scale_bounds=[1e-5, 1e5], nu=self.v_nu)
        kv_2 = ConstantKernel(constant_value=self.v_var)
        kv_3 = WhiteKernel(noise_level=self.v_noise)
        kv_4 = ConstantKernel(constant_value=self.v_mean)

        self.v_kernel = Sum(Sum(kv_4, Product(kv_1, kv_2)), kv_3)
        # Default GaussianProcessRegressor (check argument possibilities)
        self.gp_v = GaussianProcessRegressor(kernel=self.v_kernel, alpha=1e-10,
                                             optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False,
                                             copy_X_train=True, random_state=None)

        # 3 values pers sub array x, f, v
        self.delete_first_raw = False
        self.xfv = np.zeros([1, 3])

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return np.atleast_2d(self.optimize_acquisition_function())

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        # LCB acquisition (https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/acquisition.py)
        f_preds = self.gp_f.predict(np.atleast_2d(x), return_std=True)
        v_preds = self.gp_v.predict(np.atleast_2d(x))

        # Exploration - Exploitation trade - off
        acq_kappa = 0.9  # test other values
        acq_result = f_preds[0] - acq_kappa * f_preds[1]

        if (v_preds < self.kappa):
            # multiply acq_result by the weight ?
            # weight = norm(v_mu, v_std).cdf(self.kappa)
            # acq_result = acq_result * weight

            # brut penality
            acq_result = -10  # or another value (penality)

        return np.reshape(acq_result, [1])

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        # Add x, f and v to the xfv data array
        # vstack instead of concatenate ?
        xfv_new = np.append(np.append(x, np.atleast_2d(f), axis=1), np.atleast_2d(v), axis=1)
        self.xfv = np.concatenate((self.xfv, xfv_new), axis=0)
        # self.xfv = np.vstack((self.xfv, xfv_new))

        # remove first subarray of zeros used for initialization (only at first call)
        # je sais c'est bizzare mais je voulais etre sur
        if (self.delete_first_raw == False):
            self.delete_first_raw = True
            self.xfv = np.delete(self.xfv, 0, axis=0)

        # separate data for the training
        # x_train = np.atleast_2d(self.xfv[:, 0]).T
        # f_train = np.atleast_2d(self.xfv[:, 1]).T
        # v_train = np.atleast_2d(self.xfv[:, 2]).T

        x_train = np.transpose(np.atleast_2d(self.xfv[:, 0]))
        f_train = np.transpose(np.atleast_2d(self.xfv[:, 1]))
        v_train = np.transpose(np.atleast_2d(self.xfv[:, 2]))

        # Fit GPRs
        self.gp_f.fit(x_train, f_train)
        self.gp_v.fit(x_train, v_train)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # take only solution with a minimum speed
        speed_mask = (self.xfv[:, 2] >= self.kappa)

        # set of possible solutions
        solutions = self.xfv[speed_mask]

        # get the optimal solution index
        # try/except to avoid an exception on argmax with empty table
        try:
            optimal_x_idx = np.argmax(solutions[:, 1])
        except:
            optimal_x_idx = np.argmax(self.xfv[:, 1])
            solutions = self.xfv

        # return optimal solution (x)
        return solutions[optimal_x_idx, 0]


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()