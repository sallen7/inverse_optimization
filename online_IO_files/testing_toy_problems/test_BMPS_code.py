#### test_BMPS_code ####

#Need to have a test on the receive_data stuff
#from the DCZ receive_data tests - need to make sure
#BMPS is being updated properly (will need to ensure
#that the BMPS_subproblem is being updated well)

#also need to remember that receive_data for this method
#does also update the objective function with most recent c
#Is this something that should be moved??

#we can directly test the updating of the objective function 
#by feeding in the two different 
#objective functions for the chan problem.

#Look at the validation thing I wrote up for BMPS

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import copy
import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing

#Thanks to the "Managing Expressions" Page of Pyomo Documentation
#for informing me about the ability to turn pyomo
#expressions into strings
#https://pyomo.readthedocs.io/en/latest/developer_reference/expressions/managing.html
from pyomo.core.expr import current as EXPR

##### Step 1: Check that compute_standardized_model #####
## We need to check that the method creates the right standardized model ##

def test_compute_standardized_model_CLT(chan_lee_terekhov_linear_inequalities):
    chan_lee_terekhov_linear_inequalities.feasible_set_C = pyo.ConcreteModel()
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 
    model = chan_lee_terekhov_linear_inequalities.BMPS_subproblem.clone()
    
    ### Solving the Model ###
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()
    
    for (key,value) in solution.items():
        solution[key] = round(value,1)
        
    assert solution == {1:3.0,2:4.0}
    

    
def test_compute_standardized_model_quadratic_NW(quadratic_NW):
    quadratic_NW.feasible_set_C = pyo.ConcreteModel() #placeholder bc need to specify one
    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 
    model = quadratic_NW.BMPS_subproblem.clone() #clone the model
    
    ### Solving the Model ###
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    assert model.x.extract_values() == {1:2.0,2:-1.0,3:1.0} #N&W solution
    #I had <= in the Dx == f constraint generation for compute_standardized_model in the BMPS
    #methods.  Once changed to == I got that the unittests passed
    
def test_compute_standardized_model_quadratic_portfolio_Gabriel(quadratic_portfolio_Gabriel):
    quadratic_portfolio_Gabriel.feasible_set_C = pyo.ConcreteModel() #placeholder bc need to specify one
    quadratic_portfolio_Gabriel.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})  
    
    model = quadratic_portfolio_Gabriel.BMPS_subproblem.clone() #clone the model
    
    ### Solving the Model ###
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    for (key,value) in solution.items():
        solution[key] = round(value,2)
    
    assert solution == {1:0.20,2:0.44,3:0.36}

##### Step 2: Checking the project_to_F function #####

def test_project_to_F(quadratic_NW):
    #### Part 1: Defining the set C/F #####
    #TECHNICALLY HAVE A BETTER DEF FOR THE SET IN THE test_mechanics_methods
    #script
    #{1:-8,2:-3,3:-3}
    
    feasible_c_region = pyo.ConcreteModel()

    feasible_c_region.varindex = pyo.RangeSet(1,3)
    feasible_c_region.c = pyo.Var(feasible_c_region.varindex)
    
    # Defining Constraints #
    # We need all constraints to be in the form of rules (including
    # if we have non-negativity constraints)    
    def greater_than_zero(model,i):
        return 0 <= model.c[i] 
    
    feasible_c_region.greater_than_zero_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                         rule=greater_than_zero) 
    
    def less_than_one(model,i):
        return model.c[i] <= 1
    
    feasible_c_region.less_than_one_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                                rule=less_than_one)
    
    quadratic_NW.feasible_set_C = feasible_c_region.clone()
    
    #########################################################
    #We really just need to generate the model and then
    
    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})
    #the first y_t_BMPS will be the c that we passed into the initialization
    #{1:-8,2:-3,3:-3}
    quadratic_NW.next_iteration(part_for_BMPS=1) #the first y_t_BMPS is the c data that gets fed into
                                            #the class when we create an instance of it  
    
    ## Extracting the project_to_F model to check the objective function ##
    model_F = quadratic_NW.project_to_F_model.clone()
    
    ## Input 1 ##
    model_F.c[1] = 7
    model_F.c[2] = 6
    model_F.c[3] = -14
    
    obj_value_F = round(pyo.value(model_F.obj_func(model_F)),1)
    
    assert obj_value_F == 427.0
    
    ## Input 2 ##
    model_F.c[1] = -16
    model_F.c[2] = 0
    model_F.c[3] = 19
    
    obj_value_F = round(pyo.value(model_F.obj_func(model_F)),1)
    
    assert obj_value_F == 557.0
    
    ## Input 3 ##
    model_F.c[1] = -7
    model_F.c[2] = 3
    model_F.c[3] = -11
    
    obj_value_F = round(pyo.value(model_F.obj_func(model_F)),1)
    
    assert obj_value_F == 101.0
    
    
##### Step 3: solve_subproblem  

def test_solve_subproblem_CLT(chan_lee_terekhov_linear_inequalities):
    chan_lee_terekhov_linear_inequalities.feasible_set_C = pyo.ConcreteModel()
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 
    
    ### Solving the Model ###
    chan_lee_terekhov_linear_inequalities.solve_subproblem() #using solve_subproblem
    solution = copy.deepcopy(chan_lee_terekhov_linear_inequalities.xbar_t_BMPS)
    
    #(n,ph) = solution.shape
    #for i in range(0,n):
    #    solution[i,0] = round(solution[i,0],1)
    
    ### xbar_t_BMPS is a column vector ###    
    assert np.all(np.around(solution,decimals=1) == np.array([[3.0],[4.0]])) 
      
def test_solve_subproblem_quadratic_NW(quadratic_NW):
    quadratic_NW.feasible_set_C = pyo.ConcreteModel() #placeholder bc need to specify one
    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 
    
    ### Solving the Model ###
    quadratic_NW.solve_subproblem() #using solve_subproblem
    solution = copy.deepcopy(quadratic_NW.xbar_t_BMPS)
    
    ### xbar_t_BMPS is a column vector ###    
    assert np.all(np.around(solution,decimals=1) == np.array([[2.0],[-1.0],[1.0]]))
    
    
def test_solve_subproblem_quadratic_portfolio_Gabriel(quadratic_portfolio_Gabriel):
    quadratic_portfolio_Gabriel.feasible_set_C = pyo.ConcreteModel() #placeholder bc need to specify one
    quadratic_portfolio_Gabriel.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})  
    
    ### Solving the Model ###
    quadratic_portfolio_Gabriel.solve_subproblem() #using solve_subproblem
    solution = copy.deepcopy(quadratic_portfolio_Gabriel.xbar_t_BMPS)
    
    ### xbar_t_BMPS is a column vector ###    
    assert np.all(np.around(solution,decimals=2) == np.array([[0.20],[0.44],[0.36]]))

    
##### Step 4: Check gradient_step #####
#can specify stuff in the attributes and then 
#run the gradient step to make sure it did what we expected it to do
#Simple!    
    
def test_gradient_step(chan_lee_terekhov_linear_inequalities):
    ### Setting things up ###
    chan_lee_terekhov_linear_inequalities.feasible_set_C = pyo.ConcreteModel()
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 

    ### Passing in Data (to the attributes) ###
    ## Test 1 ##
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:3,2:-34}
    chan_lee_terekhov_linear_inequalities.x_t_BMPS = np.array([[10],[-24]])
    chan_lee_terekhov_linear_inequalities.xbar_t_BMPS = np.array([[16],[19]])
    eta = 0.2294
    z_t = chan_lee_terekhov_linear_inequalities.gradient_step(eta_t=eta,x_t=chan_lee_terekhov_linear_inequalities.x_t_BMPS)
    
    assert np.all(np.around(z_t,decimals=4) == np.array([[4.3764],[-24.1358]]))
    
    ## Test 2 ##
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:-5,2:-42}
    chan_lee_terekhov_linear_inequalities.x_t_BMPS = np.array([[-27],[42]])
    chan_lee_terekhov_linear_inequalities.xbar_t_BMPS = np.array([[-35],[33]])
    eta = 0.2000
    z_t = chan_lee_terekhov_linear_inequalities.gradient_step(eta_t=eta,x_t=chan_lee_terekhov_linear_inequalities.x_t_BMPS)
    
    assert np.all(np.around(z_t,decimals=4) == np.array([[-6.6000],[-43.8000]]))
    
    ## Test 3 ##
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:32,2:37}
    chan_lee_terekhov_linear_inequalities.x_t_BMPS = np.array([[-42],[-10]])
    chan_lee_terekhov_linear_inequalities.xbar_t_BMPS = np.array([[-24],[30]])
    eta = 0.1890
    z_t = chan_lee_terekhov_linear_inequalities.gradient_step(eta_t=eta,x_t=chan_lee_terekhov_linear_inequalities.x_t_BMPS)
    
    assert np.all(np.around(z_t,decimals=4) == np.array([[35.4020],[44.5600]]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    