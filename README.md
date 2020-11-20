# Bayesian Optimisation
Task 3 in the PAI course at ETHZ. We will build a model automatically optimising the best hyper-parameters for any given ML algorithm. For that, we use 

## useful links :
* [Practical BO of ML Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf) To use BO to automatically get best hyper-parameters of a model.
* [BO with unknown constraints](https://www.cs.princeton.edu/~rpa/pubs/gelbart2014constraints.pdf) To add the speed constraint to the model.
*[EI implementation](http://krasserm.github.io/2018/03/21/bayesian-optimization/)


## Ideas :
1. the objective is to maximise the function f: takes x as input (a hyper-parameter) and returns the validation accuracy on the model. 
2.At the same time, we are working on a constrained problem since we want the speed of convergence to be at least 1.2
3. We model the function f and the speed of convergence v as GPs.
4. The acquisition function "a" will give us the next hyper-parameter we should try: argmax(x){a(x)}
5. 

## methods used :
1. We used a exactGPModel to have the function "get_fantasy_model" which gives us the possibility of online learning (adding iteratively points to the GPs of f and v)
2. I used numpy's array to store new values of x, f(x) and v(x) maybe use torch?
3. Dont know what the function "next_recommendation" does.

## Next Steps:
1. Find a way to fit the speed in the model OK I THINK
2. Understand the workflow of the EI --> not compiling now
3. Find more appropriate tools --> online learning, torch, GPY? USED TORCH.TENSORs IN THE SECOND TRY
4. 

## LAST COMMIT:
1. numpy --> tensor
2. get_solutions() implemented intuitively --> check
3. add_data_point(...) missing the update on gaussian process (gp_f and gp_v) with the new data point --> need to check how to do that
4. acq_fct(...) implemented using the [gitlab from the PAI course](https://gitlab.inf.ethz.ch/scuri/pai_notebooks/-/blob/master/demos/Bayesian%20Optimization%20and%20Active%20Learning.ipynb) and taking in charge the case where the data has not yet been initiated. 
