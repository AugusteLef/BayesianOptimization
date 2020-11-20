import numpy as np
import math 
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import kernels, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, Sum, Product
from scipy.stats import norm


import torch



domain = np.array([[0, 5]])
#kernel = Matern(length_scale=0.5)

""" Solution """





class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
#         #noise perturbation sttdev
#         self.noise_nu_v          = 0.0001
#         self.noise_nu_f          = 0.15
        
#         #f's kernel
#         self.kernel_variance_f    = 0.5
#         self.kernel_lenghtscale_f = 0.5
#         self.kernel_smoothness_f  = 2.5 
        
#         #v's constant mean
#         self.meanean_v        = 1.5
#         #v's kernel
#         self.kernel_variance_v    = math.sqrt(2)
#         self.kernel_lenghtscale_v = 0.5
#         self.kernel_smoothness_v  = 2.5 
        
#         self.K = 1.2 #minimum speed K
#         #init x, v, f tensors as None as no data yet (--> add_data_point handles)
        self.x = np.array([[(np.random.rand(1) * 5)[0]]]) 
        self.v = np.array([[(np.random.rand(1))[0]]]) 
        
#         #objective function's or surrogate function's values
        self.f = np.array([[(np.random.rand(1) +1.5)[0]]]) 
    
        self.xi = 0.01
        #print(str(self.x)+"this is x init")
        #print(str(self.f)+"this is f init")
        #print(str(self.v)+"this is v init")

              
        
        
#         #self.model_v = None
#         #self.model_f = None ? ? sait aps
#         self.likelihood_f = gpytorch.likelihoods.GaussianLikelihood()
#         self.likelihood_v = gpytorch.likelihoods.GaussianLikelihood()
#         #init the kernels of f and v
        
        
#         #init the gaussian process regressors of f and v
#         #------test1-------
#         self.kernel_f = gpytorch.kernels.ScaleKernel(base_kernel = gpytorch.kernels.MaternKernel(nu=self.kernel_smoothness_f, length_scale=torch.Tensor([self.kernel_lenghtscale_f])), output_scale = self.kernel_variance_f)

#         self.kernel_v = gpytorch.kernels.ScaleKernel(base_kernel = gpytorch.kernels.MaternKernel(nu=self.kernel_smoothness_v, length_scale=self.kernel_lenghtscale_v), kernels.ConstantKernel(self.kernel_variance_v))
                                                      
#         self.gp_f = ExactGPModel(kernel = self.kernel_f, train_x = self.x, train_y = self.f, likelihood = self.likelihood_f)
#         self.gp_v = ExactGPModel(mean=self.mean_v, kernel=self.kernel_v, train_x=self.x, train_y=self.v, likelihood=self.likelihood_v)
        #------test2-------
        #self.kernel_f = Matern(nu=self.kernel_smoothness_f, length_scale=self.kernel_lenghtscale_f)* kernels.ConstantKernel(self.kernel_variance_f)

        #self.kernel_v = Matern(nu=self.kernel_smoothness_v, length_scale=self.kernel_lenghtscale_v) * kernels.ConstantKernel(self.kernel_variance_v)
        #self.gp_f = GaussianProcessRegressor(kernel = self.kernel_f, alpha = self.noise_nu_f * self.noise_nu_f)
        #self.gp_v = GaussianProcessRegressor(kernel = self.kernel_v + kernels.ConstantKernel(self.mean_v), alpha = self.noise_nu_v * self.noise_nu_v)
        
        ###self.x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
        ###         np.random.rand(domain.shape[0])
        ###self.gp_f.fit(x0, 0)
        ###self.gp_v.fit(x0, self.mean_v)
        
        #self.f = matern...
        #self.v = matern + mean...
        #self.gp = gaussian process... :p
        #ok c bon avec le ScaleKernel
        
        # TODO: enter your code here
        #
        epsilon =  1e-7
        self.kappa = 1.2 + epsilon

        # Validation Function F
        self.f_noise = 0.15
        self.f_var = 0.5
        self.f_lenscale = 0.5
        self.f_nu = 2.5

        k_1 = Matern(length_scale=self.f_lenscale, length_scale_bounds="fixed", nu=self.f_nu)
        k_2 = ConstantKernel(constant_value=self.f_var, constant_value_bounds="fixed")
        k_3 = WhiteKernel(noise_level=self.f_noise)
        self.f_kernel = Sum(Product(k_1, k_2), k_3)
        # Default GaussianProcessRegressor (check argument possibilities)
        self.gp_f = GaussianProcessRegressor(kernel=self.f_kernel)

        # Speed Function V
        self.v_noise = 0.0001
        self.v_var = np.sqrt(2)
        self.v_lenscale = 0.5
        self.v_mean = 1.5
        self.v_nu = 2.5

        kv_1 = Matern(length_scale=self.v_lenscale, length_scale_bounds="fixed", nu=self.v_nu)
        kv_2 = ConstantKernel(constant_value=self.v_var, constant_value_bounds="fixed")
        kv_3 = WhiteKernel(noise_level=self.v_noise)
        kv_4 = ConstantKernel(constant_value=self.v_mean, constant_value_bounds="fixed")

        self.v_kernel = Sum(Sum(kv_4, Product(kv_1, kv_2)), kv_3)
        # Default GaussianProcessRegressor (check argument possibilities)
        self.gp_v = GaussianProcessRegressor(kernel=self.v_kernel)


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

        #return  self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            #print(str(x)+"objectiveee")
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            #picks a rand number between [0;5] (domain of hyperparam - x -)
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            #minimize the function f (objective = (-1)*acquisitionfunction ) with x0 as initial guess using algo
            #so result[0] is the estimates x_min, result[1] is the value of f(x_min)
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            #make sure solution €[0,5] else put 0 if < 5 if >
            x_values.append(np.clip(result[0], *domain[0]))
            #append (-1)*f(x_min) to then compute the argmax.
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

        
#         if self.x == [[]]:
#             #print("CAREEE!! self.x=None")
#             return 0
        
        #mu_f, sigma_f = self.gp_f.predict(x.reshape(-1,1), return_std=True)
        #mu_v, sigma_v = self.gp_v.predict(x.reshape(-1,1), return_std=True)
        
        x_max, ymax = self.get_best_value()
        #print(ymax)
        gp_f_x  = self.gp_f.predict(x.reshape(-1,1), return_std=True)
        #print(gp_f_x)
        mean_f    = gp_f_x[0]
        sigma_f = gp_f_x[1]
            
        gp_v_x  = self.gp_v.predict(x.reshape(-1,1), return_std=True)
        mean_v  = gp_v_x[0]
        sigma_v = gp_v_x[1]
        
        
        normal_distrib = norm(torch.tensor([0.]), torch.tensor([1.]))
        #print(normal_distrib)
            
         #out = self.gp_f(self.x)
        Z = (mean_f - ymax - self.xi) / sigma_f
         #idx = out.stddev == 0
        ei_f = (mean_f - ymax - self.xi) * normal_distrib.cdf(Z) + sigma_f * normal_distrib.pdf(Z)
         #self._acquisition_function[idx] = 0 
        
        weight = norm(mean_v, sigma_v).cdf(self.kappa)
        
        return (weight * ei_f).item() 
        #------------------------------
    def get_best_value(self):
        #print(self.x)
        #print(self.f)
        idx = np.argmax(self.f)
        if len(self.f) == 1:
            xmax, ymax = self.x[idx], self.f[idx]
        else:
            #xmax, ymax = self.x[0][idx], self.f[idx]
            xmax, ymax = self.x[idx], self.f[idx]
        return xmax, ymax 
      

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
          
        #-------- test 1 -------------
        #self.x.append(x)
        #self.f.append(f)
        #self.v = torch.catappend(v)
        #self.gp_f = self.gp_f.get_fantasy_model(x, f)
        #self.gp_v = self.gp_v.get_fantasy_model(x, v)
        
        #-------- test 2 ---------------
        
        #if self.x == None:
        #    self.x = torch.Tensor(np.ndarray(x))
        #    self.f = torch.Tensor(np.ndarray(f))
        #    self.v = torch.Tensor(np.ndarray(v))
        #else: 
        #    self.x = torch.cat(self.x, x)
        #    self.f = torch.cat(self.f, f)
        #    self.v = torch.cat(self.v, v)
        #print("made it to add_data_point")
#         if self.x == [[]]:
#             self.x = np.append(self.x, x)
#             self.f = np.append(self.f, f)
#             self.v = np.append(self.v, v)
#         else:
        
        #print(type(x))
        #print(type(f))
        #print(type(v))
#         self.x = np.append(self.x, x)
        
#         self.f = np.append(self.f, f)
#         self.v = np.append(self.v, v)
        self.x = np.concatenate((self.x, x))
        self.f = np.concatenate((self.f, np.atleast_2d(f)))
        self.v = np.concatenate((self.v, np.atleast_2d(v)))
        #print(self.x)
        #print(self.f)
        #print(self.v)
        
        self.gp_f = self.gp_f.fit(self.x.reshape(-1,1), self.f.reshape(-1,1))
        self.gp_v = self.gp_v.fit(self.x.reshape(-1,1), self.v.reshape(-1,1))
        #print("leaving add_data_point")

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        sorted_f_indices = np.argsort(self.f)
        x_best_index = len(self.f)-1
        
        for index in range (len(sorted_f_indices)):
            if self.v[sorted_f_indices[index]] > 1.2:
                x_best_index = np.argmax(self.f)
        
        return self.x[x_best_index]


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
        print(j)
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
