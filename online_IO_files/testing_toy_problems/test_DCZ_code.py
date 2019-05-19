#### test_DCZ_code.py ####
#5/18/2019

#This file contains the unit tests for the Dong, Chen, & Zeng (2018)
#Online_IO algorithm methods. See the "Validation/Usage" subsection (first part)
#of the larger "Implicit Update Rule for Online Inverse Optimization" section
#of the Chapter documentation for full details/explanation regarding each test.

#The tests are labeled in terms of the (1) method they are testing and
#(2) the unit test toy problem data they are using

#We have written "Step __" headers throughout the file indicating when we
#are transitioning between testing methods.

#Additional notes:

#We utilize a MATLAB script called "generating_data.m" to obtain some
#of the test data.  See the Chapter documentation for more information.

#CLT = Chan, Lee, and Terekhov

#Different ways to call pytest:
#https://docs.pytest.org/en/latest/usage.html

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import math

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing

#Thanks to the "Managing Expressions" Page of Pyomo Documentation
#for informing me about the ability to turn pyomo
#expressions into strings
#https://pyomo.readthedocs.io/en/latest/developer_reference/expressions/managing.html
from pyomo.core.expr import current as EXPR


#### Step 1: Check the compute_KKT_conditions method ####
## We will use the unit test toy problems to check that the 
## compute_KKT_conditions method is doing its job
    
def test_compute_KKT_conditions_CLT(chan_lee_terekhov_linear_inequalities):
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()
    
    #####################################################
    #model.obj_func = pyo.Objective(expr=5)    
    
    #### Dummy variable for objective function #####
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    #################################################
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()
    
    for (key,value) in solution.items():
        solution[key] = round(value,1)
    
    assert model.x.extract_values() == {1:3.0,2:4.0}
    
def test_compute_KKT_conditions_CLT_non_zero(chan_lee_terekhov_linear_inequalities):
    chan_lee_terekhov_linear_inequalities.non_negative = 1 #change to 1
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()
    
    #####################################################
    #model.obj_func = pyo.Objective(expr=5) 
    
    #### Dummy Variable for Objective Function ####
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)    

    #################################################
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()
    
    for (key,value) in solution.items():
        solution[key] = round(value,1)
    
    assert model.x.extract_values() == {1:3.0,2:4.0}


def test_compute_KKT_conditions_quadratic_NW(quadratic_NW):
    quadratic_NW.initialize_IO_method("Dong_implicit_update")
    
    model = quadratic_NW.KKT_conditions_model.clone()
    
    #####################################################
    #model.obj_func = pyo.Objective(expr=5)    
    
    ### Dummy Variable for Objective Function ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    #################################################
    
    solver = SolverFactory("gurobi") 
        
    solver.solve(model)
    
    assert model.x.extract_values() == {1:2.0,2:-1.0,3:1.0}
    
    assert model.v.extract_values() == {1:-3.0,2:2.0} #{1:3.0,2:-2.0}
    #had to negate the given lambda because N&W Lagrangian assumes a 
    #slightly different form where the lambdas are subtracted    
    
    
def test_compute_KKT_conditions_quadratic_portfolio_Gabriel(quadratic_portfolio_Gabriel):
    quadratic_portfolio_Gabriel.initialize_IO_method("Dong_implicit_update") #initialize the method
      
    model = quadratic_portfolio_Gabriel.KKT_conditions_model.clone() 
    
    ## Dummy Variable for Objective ##
    #model.obj_func = pyo.Objective(expr=5)
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    #Solve the model
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    for (key,value) in solution.items():
        solution[key] = round(value,2)
    
    assert solution == {1:0.20,2:0.44,3:0.36}
    
#### Step 2: Examine loss_function ####
# Mainly want to check the objective function, but also
# have a small test problem 
    
def test_loss_function_CLT_test_problem(chan_lee_terekhov_linear_inequalities):
    ## Part a: Create the KKT_model ##
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    
    ## Part b: Call loss_function ##
    y_t = np.array([[2.5],[3]])
    c_t = {1:2,2:-3}
    chan_lee_terekhov_linear_inequalities.dong_iteration_num = 1
    
    output = chan_lee_terekhov_linear_inequalities.loss_function(y=y_t,theta=c_t)
    
    ## Part c: Extract Solution ##
    solution = chan_lee_terekhov_linear_inequalities.loss_model_dong.x.extract_values()
    for (key,value) in solution.items():
        solution[key] = round(value,2)
    
    assert solution == {1:2.19,2:3.46}
    
    ## Part d: Check the c variable got set correctly ##
    assert chan_lee_terekhov_linear_inequalities.loss_model_dong.c.extract_values() == c_t
    for i in range(1,3):
        assert chan_lee_terekhov_linear_inequalities.loss_model_dong.c[i].fixed == True #make sure variables still
                #fixed
    
    
def test_loss_function_obj_func(chan_lee_terekhov_linear_inequalities):
    ### NOW we want to just do a few checks of the objective function ###
    
    ## Part a: Create the KKT_model ##
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    
    ## Part b: Call loss_function ##
    y_t = np.array([[2.5],[3]])
    c_t = {1:2,2:-3}
    chan_lee_terekhov_linear_inequalities.dong_iteration_num = 1
    
    output = chan_lee_terekhov_linear_inequalities.loss_function(y=y_t,theta=c_t)
    
    ## Putting Model in Variable ##
    model_loss = chan_lee_terekhov_linear_inequalities.loss_model_dong.clone()
    
    ## Now we will input different x values to ensure that we are obtaining
    ## the objective function we desire
    
    ## Test 1 ##
    
    model_loss.x[1] = 32
    model_loss.x[2] = 41
        
    obj_value_loss = round(pyo.value(model_loss.obj_func(model_loss)),2)
    
    assert obj_value_loss == 2314.25 #had to go out to 2 decimal places to get
    #things to work out
    
    ## Test 2 ##
    model_loss.x[1] = -38
    model_loss.x[2] = 42
    
    obj_value_loss = round(pyo.value(model_loss.obj_func(model_loss)),2)
    
    assert obj_value_loss == 3161.25 #had to go out to 2 decimal places to get
    #things to work out
    
    ## Test 3 ##
    model_loss.x[1] = -22
    model_loss.x[2] = 5
    
    obj_value_loss = round(pyo.value(model_loss.obj_func(model_loss)),2)
    
    assert obj_value_loss == 604.25    

#### Step 3: Examine the update_rule_optimization_model ####
    
def test_update_rule_optimization_model_obj_func(chan_lee_terekhov_linear_inequalities):
    chan_lee_terekhov_linear_inequalities.feasible_set_C = (-10,100) #putting bounds
    
    ## Part a: Create the KKT_model ##
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    
    ## Part b: Call the update_rule_optimization_model ##
    #c and x are the variables
    #need to pass in c_t, y, and eta_t
    y_t = np.array([[2.5],[3]])
    c_t = {1:2,2:-3}
    eta_t = 5*(1/math.sqrt(30)) #with next_iteration, a user would just pass in the 5 on the 30th iteration
                                #to the eta_factor argument.  Here though, we are working with
                                #update_rule_optimization_model directly, so we have to pass in 
                                #eta_t entirely
    chan_lee_terekhov_linear_inequalities.dong_iteration_num = 1
    
    chan_lee_terekhov_linear_inequalities.update_rule_optimization_model(y=y_t,\
                                                theta=c_t,eta_t=eta_t)
    
    model_update = chan_lee_terekhov_linear_inequalities.update_model_dong.clone()   
    
    #### Check that the c variables were unfixed ####
    for i in range(1,3):
        assert chan_lee_terekhov_linear_inequalities.update_model_dong.c[i].fixed == False #make sure variables still
                #fixed
    
    #### Check that the lower and upper bound constraints were constructed ####
    assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_lower._constructed == True
    assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_upper._constructed == True
    
    for i in range(1,3):
        assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_lower[i].lower == -10
        assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_lower[i].upper == None
        assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_upper[i].upper == 100
        assert chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_upper[i].lower == None
    
    for i in range(1,3):
        chan_lee_terekhov_linear_inequalities.update_model_dong.c[i] = -20
        assert pyo.value(chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_lower[i].body) == -20
        assert pyo.value(chan_lee_terekhov_linear_inequalities.update_model_dong.c_bound_upper[i].body) == -20
        
    
    #### Check the Objective Function ####
    ## Test 1 ##
    model_update.x[1] = -23
    model_update.x[2] = 25
    
    model_update.c[1] = 19
    model_update.c[2] = 25
    
    obj_value_update = round(pyo.value(model_update.obj_func(model_update)),1)
    
    assert obj_value_update == 1571.9
    
    ## Test 2 ##
    model_update.x[1] = -14
    model_update.x[2] = 3
    
    model_update.c[1] = 8
    model_update.c[2] = -25
    
    obj_value_update = round(pyo.value(model_update.obj_func(model_update)),2)
    
    assert obj_value_update == 508.53
    
    ## Test 3 ##
    model_update.x[1] = -21
    model_update.x[2] = 29
    
    model_update.c[1] = 28
    model_update.c[2] = 28
    
    obj_value_update = round(pyo.value(model_update.obj_func(model_update)),1)
    
    assert obj_value_update == 1939.7
