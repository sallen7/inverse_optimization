## test_mechanics_methods ##

#We want to move tests that focus mainly on the mechanics methods of the 
#Online_IO class to this file.

#Think I also am going to be really ONLY focusing on the receive_data method
#and testing that with many different things because the other methods
#are tested via the experiment done for each of the math methods
#I will say that the other methods just call other methods within the class
#to carry out the execution of stuff (basically, will check...)

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing


##### Part 1: Testing the receive_data method #####
# We will need to test the receive data method for 
# both the DCZ and BMPS algorithms, since receive_data contains
# an outer if/else that depends upon these which algorithm we initiated

def test_receive_data_quadratic_NW_DCZ(quadratic_NW):    
    quadratic_NW.initialize_IO_method("Dong_implicit_update")
    
    ### Test 1 ###
    
    A_1 = {(1,1):-17, (1,2):0, (1,3):-18, (2,1):0, (2,2):1, (2,3):11}
    b_1 = {1:18, 2:0} #pretty sure single indexing for bs
    
    quadratic_NW.receive_data(p_t={'Amatrix':A_1, 'bvector':b_1}) #should update
                                    #the parameters accordingly
    
    model = quadratic_NW.KKT_conditions_model.clone()
    model.obj_func = pyo.Objective(expr=5)  #add a constant objective func 
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    dual_vars = model.v.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
    for (key,value) in dual_vars.items():
        dual_vars[key] = round(value,4)
        
    assert solution == {1:-0.9430,2:1.2038,3:-0.1094} 
    #assert dual_vars == {1:0.6682,2:0.9141}
    
    ### Test 2 ###
    
    A_2 = {(1,1):-15, (1,2):0, (1,3):3, (2,1):0, (2,2):-1, (2,3):-20}
    b_2 = {1:-7, 2:0} #pretty sure single indexing for bs
    
    quadratic_NW.receive_data(p_t={'Amatrix':A_2, 'bvector':b_2}) #should update
                                    #the parameters accordingly
    
    model = quadratic_NW.KKT_conditions_model.clone()
    model.obj_func = pyo.Objective(expr=5)  #add a constant objective func 
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    dual_vars = model.v.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
    for (key,value) in dual_vars.items():
        dual_vars[key] = round(value,4)
        
    assert solution == {1:0.4627,2:0.3957,3:-0.0198} 
    #assert dual_vars = {1:0.2968,2:0.1358} #here is a minus sign thing 
    #I need to check    

def test_receive_data_quadratic_NW_BMPS(quadratic_NW):
    ### Generating the C/F Feasible Region for Projection Purposes ###
    #I COULD HAVE ALSO JUST FED IN THE CORRECT VECTOR TO c_t_BMPS....
    feasible_c_region = pyo.ConcreteModel()

    feasible_c_region.varindex = pyo.RangeSet(1,3)
    feasible_c_region.c = pyo.Var(feasible_c_region.varindex)
    
    # Defining Constraints #
    # We need all constraints to be in the form of rules (including
    # if we have non-negativity constraints)    
    def greater_than_negative_ten(model,i):
        return -10 <= model.c[i] 
    
    feasible_c_region.greater_than_negative_ten_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                         rule=greater_than_negative_ten) 
    
    def less_than_one(model,i):
        return model.c[i] <= 1
    
    feasible_c_region.less_than_one_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                                rule=less_than_one)
    
    quadratic_NW.feasible_set_C = feasible_c_region.clone()


    ###########################################################################
    #### Getting Back to the where can test receive_data ####
    quadratic_NW.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})
    quadratic_NW.next_iteration(part_for_BMPS=1) #to generate the c_t_BMPS 
    
    ### Test 1 ###
    
    A_1 = {(1,1):-17, (1,2):0, (1,3):-18, (2,1):0, (2,2):1, (2,3):11}
    b_1 = {1:18, 2:0} #pretty sure single indexing for bs
    
    ## Difference for BMPS is that the objective function for BMPS_subproblem
    ## will be updated as well
    quadratic_NW.receive_data(p_t={'Amatrix':A_1, 'bvector':b_1}) #should update
                                    #the parameters accordingly
    
    model = quadratic_NW.BMPS_subproblem.clone()
       
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    #dual_vars = model.v.extract_values() #BMPS_subproblem does not HAVE dual variables

    for (key,value) in solution.items():
        solution[key] = round(value,4)
#    for (key,value) in dual_vars.items():
#        dual_vars[key] = round(value,4)
        
    assert solution == {1:-0.9430,2:1.2038,3:-0.1094} 
    #assert dual_vars == {1:0.6682,2:0.9141}
    
    ### Test 2 ###
    
    A_2 = {(1,1):-15, (1,2):0, (1,3):3, (2,1):0, (2,2):-1, (2,3):-20}
    b_2 = {1:-7, 2:0} #pretty sure single indexing for bs
    
    quadratic_NW.receive_data(p_t={'Amatrix':A_2, 'bvector':b_2}) #should update
                                    #the parameters accordingly
    
    model = quadratic_NW.BMPS_subproblem.clone()
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    #dual_vars = model.v.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
#    for (key,value) in dual_vars.items():
#        dual_vars[key] = round(value,4)
        
    assert solution == {1:0.4627,2:0.3957,3:-0.0198} 
    #assert dual_vars = {1:0.2968,2:0.1358} #here is a minus sign thing 
    #I need to check    



def test_receive_data_CLT_DCZ():
    
    assert 3 == 3


def test_receive_data_CLT_BMPS():
    #Can directly pass in different c values into c_t_BMPS to check that 
    #the c update step is working - can pass in the EXACT SAME
    #data for p_t and, before this, you could have changed the c_t_BMPS (look at style of this 
    #constraint again) 
    
    #Need to do CLT with and without x >= 0 bc we need to check
    #the regeneration of expressions
    
    #Want to re-read the thread about reconstruct as a double check
    
    
    assert 3 == 3





