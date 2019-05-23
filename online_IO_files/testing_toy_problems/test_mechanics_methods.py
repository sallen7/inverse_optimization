## test_mechanics_methods ##
# 5/18/2019

#This test file contains the unit tests that are associated with the mechanics
#methods of the Online_IO class.  As noted in the Chapter documentation, we will
#be focusing on testing the receive_data method.

#Readers can find more information regarding these tests in the "Validation"
#subsection of the "Overview of Online Methods and the Online_IO Class" section
#of the Chapter documentation.  We provide detailed explanations of all
#the tests by name, so that is the place to go for explanations.

#The names of the tests are meant to convey the (1) unit test toy problem
#we are using and the (2) online inverse optimization method we are using
#which is either DCZ or BMPS (initials of the researchers)

#We took the methods descriptions from the Chapter documentation so, for any
#citations, see that document.

#Additional Notes:

#We utilize a MATLAB script called "generating_data.m" to obtain some
#of the test data.  See the Chapter documentation for more information.

#CLT = Chan, Lee, and Terekhov

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
    #TEST DESCRIPTION: This unit test checks that the KKT_conditions_
    #model pyomo model for the Dong, Chen, & Zeng [16] algorithm is updated correctly when
    #we pass in two different configurations of the equality constraints for the second unit
    #test toy problem (“Quadratic Program with Equality Constraints”).
    
    quadratic_NW.initialize_IO_method("Dong_implicit_update")
    
    ### Test 1 ###
    
    A_1 = {(1,1):-17, (1,2):0, (1,3):-18, (2,1):0, (2,2):1, (2,3):11}
    b_1 = {1:18, 2:0} #pretty sure single indexing for bs
    
    quadratic_NW.receive_data(p_t={'Amatrix':A_1, 'bvector':b_1}) #should update
                                    #the parameters accordingly
    
    model = quadratic_NW.KKT_conditions_model.clone()
    ## Dummy Variable ##
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    #model.obj_func = pyo.Objective(expr=5)  #add a constant objective func 
    
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
                                    
    #print("Quadratic_NW DCZ:")
    #quadratic_NW.KKT_conditions_model.pprint()
    
    model = quadratic_NW.KKT_conditions_model.clone()
    
    ### Dummy Variable Method ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    
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
        

def test_receive_data_quadratic_NW_BMPS(quadratic_NW):
    #TEST DESCRIPTION: This unit test checks that the BMPS_subproblem
    #pyomo model for the Barmann, Martin, Pokutta, and Schneider [3] 
    #algorithm is updated correctly when we pass in two different 
    #configurations of the equality constraints
    #for the second unit test toy problem (“Quadratic Program with Equality Constraints”)
    
    
    ###############################################################
    ### Generating the C/F Feasible Region for Projection Purposes ###
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
    b_1 = {1:18, 2:0} #single indexing for bs
    
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
        
    assert solution == {1:-0.9430,2:1.2038,3:-0.1094} 
    
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

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:0.4627,2:0.3957,3:-0.0198} 


def test_receive_data_CLT_DCZ(chan_lee_terekhov_linear_inequalities):
    #TEST DESCRIPTION: This unit test checks that the KKT_conditions_model
    #pyomo model for the Dong, Chen, & Zeng [16] algorithm is updated correctly when
    #we change the (4) constraint for unit test toy problem 1 twice.
    
    
    ### Test Receiving Data ### 
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    #print("chan_lee_terekhov_linear_inequalities (right after initialize_IO_method):")
    #chan_lee_terekhov_linear_inequalities.KKT_conditions_model.pprint()
    
    ### Passing Data: Test 1 ###
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.9, (4,2):1}
    
    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()
    
    ### Dummy Variable Method ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    #model.obj_func = pyo.Objective(expr=5)  #add a constant objective func 
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)        
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.1169,2:4.0779}
    
    ### Passing Data: Test 2 ###
    # 3.2432     4.1622
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.8, (4,2):1}
    #2,5],[2,-3],[2,1],[-2,-1
    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    #print("chan_lee_terekhov_linear_inequalities: (after reconstruct 2)")                                
    #chan_lee_terekhov_linear_inequalities.KKT_conditions_model.pprint()
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()

    ### Dummy Variable Method ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.2432,2:4.1622} #we show that the solution changes!!
    #data was updated correctly!!
    
    

def test_receive_data_CLT_DCZ_non_negative(chan_lee_terekhov_linear_inequalities):
    #TEST DESCRIPTION: Same as the previous one, except we add
    #in the constraint x >= 0, which doesn’t change the solutions but does affect how the
    #KKT_conditions_model is formed and thus updated.
    
    
    ### Changing the Non-negative Flag to 1 ###
    chan_lee_terekhov_linear_inequalities.non_negative = 1    
    
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("Dong_implicit_update")
    print("chan_lee_terekhov_linear_inequalities NON NEG - right after initialize_IO_method:")
    chan_lee_terekhov_linear_inequalities.KKT_conditions_model.pprint()
    
    ### Passing Data: Test 1 ###
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.9, (4,2):1}
    
    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()
    
    ### Dummy Variable Method ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.1169,2:4.0779}
    
    ### Passing Data: Test 2 ###
    # 3.2432     4.1622
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.8, (4,2):1}
    
    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    print("chan_lee_terekhov_linear_inequalities NON NEGATIVE - after receive data 2")
    chan_lee_terekhov_linear_inequalities.KKT_conditions_model.pprint()
    
    model = chan_lee_terekhov_linear_inequalities.KKT_conditions_model.clone()
 
    ### Dummy Variable Method ###
    model.alpha = pyo.Var([1])
    model.alpha_constraint = pyo.Constraint(expr=model.alpha[1]==5)
    model.obj_func = pyo.Objective(expr = model.alpha[1]*3)
    
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.2432,2:4.1622}
    


def test_receive_data_CLT_BMPS(chan_lee_terekhov_linear_inequalities):    
    # TEST DESCRIPTION: This unit test checks that the BMPS_subproblem pyomo
    # model for the Barmann, Martin, Pokutta, and Schneider [3] algorithm is updated correctly
    # when we change the (4) constraint for unit test toy problem 1 twice.
    
    
    ## Putting values in parameters when need to ##
    chan_lee_terekhov_linear_inequalities.feasible_set_C = pyo.ConcreteModel()
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:(-1),2:(-1)}
    
    ## Initializing the Method ##
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})
    
    ########################################################
    ### Passing Data: Test 1 ###
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.9, (4,2):1}

    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    model = chan_lee_terekhov_linear_inequalities.BMPS_subproblem.clone()
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.1169,2:4.0779}
    
    ### Passing Data: Test 2 ###
    # 3.2432     4.1622
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):1.8, (4,2):1}

    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1, 'bvec':b_1}) #should update
                                    #the parameters accordingly
    
    model = chan_lee_terekhov_linear_inequalities.BMPS_subproblem.clone()
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,4)
        
    assert solution == {1:3.2432,2:4.1622}
    
    
    
def test_receive_data_diff_c_CLT_BMPS(chan_lee_terekhov_linear_inequalities):
    #TEST DESCRIPTION: This unit test checks that the BMPS_subproblem’s
    #objective function is updated with the most recent ct guess.  We artifically 
    #pass c_ts to the c_t_BMPS attribute
    
    #Can directly pass in different c values into c_t_BMPS to check that 
    #the c update step is working - can pass in the EXACT SAME
    #data for p_t and, before this, you could have changed the c_t_BMPS (look at style of this 
    #constraint again) 
    
    chan_lee_terekhov_linear_inequalities.feasible_set_C = pyo.ConcreteModel()
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:(-1),2:(-1)}
    
    ## Initializing the Method ##
    chan_lee_terekhov_linear_inequalities.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0})
    
    ### Test 1: c = (-1,-1) ###
    #We will NOT be changing the feasible region, only will be changing the 
    #c data between the following two tests
    #We do still need to pass data in because we IDed A and b as being mutable (will just be the same data)
    
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):2, (4,2):1}

    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1,'bvec':b_1})
    
    model = chan_lee_terekhov_linear_inequalities.BMPS_subproblem.clone()
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,1)
        
    assert solution == {1:3.0,2:4.0}
    
    
    ### Test 2: c = (3/5,-2/5) ###
    chan_lee_terekhov_linear_inequalities.c_t_BMPS = {1:(3/5),2:(-2/5)} #change the c
    
    A_1 = {(1,1):-2, (1,2):-5, (2,1):-2, (2,2):3, (3,1):-2, (3,2):-1, (4,1):2, (4,2):1}

    b_1 = {1:-10,2:6,3:-4,4:10} #pretty sure single indexing for bs
    
    chan_lee_terekhov_linear_inequalities.receive_data(p_t={'Amat':A_1,'bvec':b_1})
    
    model = chan_lee_terekhov_linear_inequalities.BMPS_subproblem.clone()
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    solution = model.x.extract_values()

    for (key,value) in solution.items():
        solution[key] = round(value,2)
    
    assert solution == {1:0.75,2:2.50}




