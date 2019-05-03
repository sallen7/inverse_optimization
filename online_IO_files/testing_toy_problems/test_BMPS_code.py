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
    
def test_compute_standardized_model_quadratic_NW(quadratic_NW):
    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #this will set up the method 
    model = quadratic_NW.BMPS_subproblem.clone() #clone the model
    
    ### Solving the Model ###
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    assert model.x.extract_values() == {1:2.0,2:-1.0,3:1.0} #N&W solution
    #I had <= in the Dx == f constraint generation for compute_standardized_model in the BMPS
    #methods.  Once changed to == I got that the unittests passed
    
def test_compute_standardized_model_quadratic_portfolio_Gabriel(quadratic_portfolio_Gabriel):
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
    
##NEED TO ADD THE CHAN EXAMPLE/EXAMPLES TO THIS (BUT NEED TO UNTANGLE THEM)

##### Step 2: Checking the project_to_F function #####

#def test_project_to_F_quadratic_NW(quadratic_NW):
#    #### Part 1: Defining the set C/F #####
#    #TECHNICALLY HAVE A BETTER DEF FOR THE SET IN THE test_mechanics_methods
#    #script
#    
#    feasible_c_region = pyo.ConcreteModel()
#
#    feasible_c_region.varindex = pyo.RangeSet(1,3)
#    feasible_c_region.c = pyo.Var(feasible_c_region.varindex)
#    
#    # Defining Constraints #
#    # We need all constraints to be in the form of rules (including
#    # if we have non-negativity constraints)    
#    def greater_than_zero(model,i):
#        return 0 <= model.c[i] 
#    
#    feasible_c_region.greater_than_zero_constraint = pyo.Constraint(feasible_c_region.varindex,\
#                                                         rule=greater_than_zero) 
#    
#    def less_than_one(model,i):
#        return model.c[i] <= 1
#    
#    feasible_c_region.less_than_one_constraint = pyo.Constraint(feasible_c_region.varindex,\
#                                                                rule=less_than_one)
#    
#    quadratic_NW.feasible_set_C = feasible_c_region.clone()
#    
#    #########################################################
#    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})
#    quadratic_NW.next_iteration(part_for_BMPS=1) #the first y_t_BMPS is the c data that gets fed into
#                                            #the class when we create an instance of it  
    
    
    
    ##################################################
    #Just need to pass values to x and then evaluate the objective function like below
    #MIGHT also think about an easy project-to-set example (see the proximal stuff)
    
    #quadratic_NW.project_to_F_model.
    
#    (Pdb) toyprob1.x[1] = 4
#    (Pdb) toyprob1.constraint1.rule(toyprob1)
#    <pyomo.core.expr.expr_pyomo5.InequalityExpression object at 0x0000016207681BA8>
#    (Pdb) toyprob1.constraint1.rule(toyprob1).value
#    *** AttributeError: 'InequalityExpression' object has no attribute 'value'
#    (Pdb) pyo.value(toyprob1.constraint1.rule(toyprob1))
#    ERROR: evaluating object as numeric value: x[2]
#        (object: <class 'pyomo.core.base.var._GeneralVarData'>)
#    No value for uninitialized NumericValue object x[2]
#    ERROR: evaluating object as numeric value: b_param[1]  <=  x[1] - x[2]
#        (object: <class 'pyomo.core.expr.expr_pyomo5.InequalityExpression'>)
#    No value for uninitialized NumericValue object x[2]
#    *** ValueError: No value for uninitialized NumericValue object x[2]
#    (Pdb) toyprob1.x[2] = 4
#    (Pdb) pyo.value(toyprob1.constraint1.rule(toyprob1))
#    False
#    (Pdb) pyo.value(toyprob1.obj_func(toyprob1))
#    8
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    