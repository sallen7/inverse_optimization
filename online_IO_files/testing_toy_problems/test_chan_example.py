#### Unit Test File for the Chan Example ####

#Different ways to call pytest:
#https://docs.pytest.org/en/latest/usage.html

#Need to remember that we are just testing
#mechanics of the model; we don't need to 
#come up with this elaborate "answer we are trying
#to get to through multiple iterations"

#BUT, saying that, do we want to give all of the
#exact data (c,x0,etc)

#WHERE LEFT OFF: Really need to do a workflow track tomorrow (of like
#all the steps to take something from start to finish) and 
#write up more documentation

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


#NEED A TEST METHOD THAT CHECKS THE INITIALIZATION
#OF THE INSTANCE

def test_initialize_IO_method(chan_init_IO_method):
    #Step 1: Ensure .alg_specification attribute
    #is updated correctly
    assert chan_init_IO_method.alg_specification == "Dong_implicit_update"
    #Step 2: Make sure the c_t_dong attribute
    #is initialized correctly (with the data
    #from the c parameter)
    assert chan_init_IO_method.c_t_dong == {1:(2/5),2:(-3/5)}
    #Step 3: Checking the KKT_conditions_model
    #THINK THE BETTER WAY TO DO THIS IS TO JUST use
    #test problems that directly test the compute_KKT_conditions method
    
    #KKT_model = chan_init_IO_method.KKT_conditions_model
    #A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    #A_t = np.transpose(A)
    #testing_expression = test_model.Abconstraints[1].body
    #string_rep = EXPR.expression_to_string(testing_expression)
    #comparison_thing = "-2*x[1] - 5*x[2]"
    #for i in range(1,5):
    #    KKT_expression = KKT_model.stationary_conditions[i].body
        #NEED TO LOOP THROUGH THE CONSTRAINTS AND 
        #CREATE string EXPRESSIONS THAT MATCH THE CONSTRAINTS
        #might want to look at regular expressions
        #Might want to also write up some of the KKT math
        #while you are writing up this code
    
def test_chan_KKT_conditions_generate_1(chan_compute_KKT_1):
    #We caught an issue/oversight in our code due to this test
    #In the compute_KKT_conditions method, we now set the fixed
    #c values to the c data that comes into the code
    model = chan_compute_KKT_1.KKT_conditions_model
    #pdb.set_trace()
    
    solver = SolverFactory("gurobi") 
        
    solver.solve(model)
    ##Step 1: Assert that Solved to optimality##
    #CANT DO THIS BECAUSE CONSTANT OBJECTIVE!!
    #Need to talk about why you can do this in the report
    #assert results.solver.termination_condition == "optimal"

    ##Step 2: Assert that we obtained the correct answer
    assert model.x.extract_values() == {1:3.0,2:4.0}

def test_chan_KKT_conditions_generate_2(chan_compute_KKT_2):
    model = chan_compute_KKT_2.KKT_conditions_model
    solver = SolverFactory("gurobi") 
        
    solver.solve(model)
    ##Step 1: Assert that Solved to optimality##
    #CANT DO THIS BECAUSE CONSTANT OBJECTIVE!!
    #assert results.solver.termination_condition == "optimal"

    ##Step 2: Assert that we obtained the correct answer
    assert model.x.extract_values() == {1:0.75,2:2.50}     
    
    
def test_receive_data(chan_receive_data):
    ##POTENTIALLY ADD MORE STUFF!!
    ##Step 2: Check that the ``noisy observation''
    ##gets put into the correct attribute
    assert np.all(chan_receive_data.noisy_decision_dong == np.array([[2.5],[3]])) 
    #NEED TO FIX -  ValueError: The truth value 
    #of an array with more than one element is ambiguous. Use a.any() or a.all()
    
    
def test_that_returns_fail(chan_init_instance):    
    ##This test will ensure that the receive_data
    ##and next_iteration methods will not
    ##execute and will return error when we attempt
    ##to run them before the initialize_IO_method
    ##method
    
    #CAN I ASSERT THAT SOMETHING PRINTS??
    
    ##Step 1: receive_data method
    assert chan_init_instance.receive_data(x_t=5) == 0 #make sure returns a 0
    #Ensure that noisy decision was not updated
    assert chan_init_instance.noisy_decision_dong is None
    
    ##Step 2: next_iteration() method
    assert chan_init_instance.next_iteration() == 0 #make sure returns a 0
    #Ensure the iteration number hasn't increased, losses_dong list
    #is still an empty list, and c_t_dong is still None
    assert chan_init_instance.dong_iteration_num == 1
    assert not chan_init_instance.losses_dong #assert that it is still 
                                            #empty (that no losses have been 
                                            #appended to it)
    assert chan_init_instance.c_t_dong is None
    
    
    
    
###FOR THE update rule test, we still need to see if we should 
#change the unfixed c values from their current values
    
    
    
    
    
    
