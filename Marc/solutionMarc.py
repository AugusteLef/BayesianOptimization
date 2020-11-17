import numpy as np
import math 
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import kernels
from scipy.stats import norm

domain = np.array([[0, 5]])
kernel = Matern(length_scale=0.5)

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.nu_v = 0.0001
        self.mean_v = 1.5
        self.nu_f = 0.15
        self.K = 1.2
        self.x = None
        self.v = None
        self.f = None
        
        
        #self.model_v = None
        #self.model_f = None ? ? sait aps
        self.likelihood_f = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood_v = gpytorch.likelihoods.GaussianLikelihood()
        
        self.kernel_f = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=torch.Tensor([0.5])), outputscale=0.5)

        self.kernel_v = gpytorch.kernels.ScaleKernel(base_kernel=gpytorch.kernels.MaternKernel(nu=2.5, lengthscale=torch.Tensor([0.5])), outputscale=np.sqrt(2))
       
        self.gp_f = ExactGPModel(kernel = self.kernel_f, train_x = self.x, train_y = self.f, likelihood = self.likelihood_f)
    
        self.gp_v = ExactGPModel(mean=self.mean_v, kernel=self.kernel_v, train_x=self.x, train_y=self.v, likelihood=self.likelihood_v)
        
        
        #self.f = matern...
        #self.v = matern + mean...
        #self.gp = gaussian process... :p
        #ok c bon avec le ScaleKernel
        
        


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
        return self.optimize_acquisition_function(self)


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
        #return acquisition_function_PI(self, x)
        return acquisition_function_EI(self, x)
        #return acquisition_function_LCB(self, x)
        
        #raise NotImplementedError
    def acquisition_function_EI(self, x):
        """
        Probability of Improvement acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the Probability of Improvement at x
        """
        #--------- test 1 ----------
        #fx_estimate = self.gp_f(gp_f, x[len(x)-1])
        
        #z = (np.mean(x) - fx_estimate ) / math.sqrt(np.var(x))
        
        #ei = (np.mean(x) - fx_estimate) * norm.cdf(fx_estimate)  + math.sqrt(np.var(x)) * norm.pdf(fx_estimate)
        #return ei
        #--------- test 2 ----------- (USING https://gitlab.inf.ethz.ch/scuri/pai_notebooks/-/blob/master/demos/Bayesian%20Optimization%20and%20Active%20Learning.ipynb
        
        self.xmax, self.ymax = self.get_best_value()
        
        if self.gp_f.train_inputs == None:
            mu_f    = 2.5
            sigma_f = math.sqrt(0.5)
            mu_v    = 1.5
            sigma_v = math.sqrt(math.sqrt(2))
        else:
            x      = torch.Tensor(x)
            
            gp_f_x  = self.gp_f(x)
            mu_f    = gp_f_x.mean
            sigma_f = gp_f_x.stddev
            
            gp_v_x  = self.gp_v(x)
            mu_v  = gp_v_x.mean
            sigma_v = gp_v_x.stddev
        
        
        normal_distrib = Normal(torch.tensor([0.]), torch.tensor([1.]))

            
        #out = self.gp_f(self.x)
        Z = (out.mean - ymax - self.xi) / sigma_f
        #idx = out.stddev == 0
        ei_f = (mu_f - self.ymax - self.xi) * normal_distrib.cdf(Z) + sigma_f * torch.exp(normal_distrib.log_prob(Z))
        #self._acquisition_function[idx] = 0 
        
        weight = Normal(mu_v, sigma_v).cdf(self.K)
        
        return weight * ei_f  
        #------------------------------
      
        
    #----for the test 2 in acqFunc----
    def get_best_value(self):
        idx = self.gp.train_targets.argmax()
        if len(self.gp.train_targets) == 1:
            xmax, ymax = self.gp.train_inputs[idx], self.gp.train_targets[idx]
        else:
            xmax, ymax = self.gp.train_inputs[0][idx], self.gp.train_targets[idx]
        return xmax, ymax 
    #---------------------------------

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
        
        if self.x == None:
            self.x = torch.Tensor(x)
            self.f = torch.Tensor(f)
            self.v = torch.Tensor(v)
        else: 
            self.x = torch.cat(self.x, x)
            self.f = torch.cat(self.f, f)
            self.v = torch.cat(self.v, v)
         
        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        sorted_f_indices = np.argsort(f)
        x_best_index = len(f)-
        
        for index in range (len(sorted_f_indices.f)):
            if v[sorted_f_indices[index]] > 1.2:
                x_best_index = np.argmax(self.f)
       
        return x[x_best_index]


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