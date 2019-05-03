#### Unit Test File for the Chan Example ####

#Different ways to call pytest:
#https://docs.pytest.org/en/latest/usage.html

#Need to remember that we are just testing
#mechanics of the model; we don't need to 
#come up with this elaborate "answer we are trying
#to get to through multiple iterations"

#BUT, saying that, do we want to give all of the
#exact data (c,x0,etc)

#NEW COMMENT: 5/3/2019 Many of the Chan tests are not extremely useful
#or at least could be combined more (and definitely don't need a million
#fixtures defined to implement them)

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
    model = chan_compute_KKT_1.KKT_conditions_model.clone()
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
    model = chan_compute_KKT_2.KKT_conditions_model.clone()
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
    assert chan_init_instance.dong_iteration_num == 0
    assert not chan_init_instance.losses_dong #assert that it is still 
                                            #empty (that no losses have been 
                                            #appended to it)
                                            #https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty
    assert chan_init_instance.c_t_dong is None
    
def test_NW_quadratic_KKT(quadratic_NW):
    quadratic_NW.initialize_IO_method("Dong_implicit_update")
    
    model = quadratic_NW.KKT_conditions_model.clone()
    
    #####################################################
    # NEW CODE 4/29/2019: I'm putting the objective function in the
    # test because want to be able to use quadratic_NW for multiple
    # tests
    ### Adding Objective Function: As long as Satisfy KKT N&S then
    ## Can use Constant Objective Function 
    model.obj_func = pyo.Objective(expr=5)    
    
    #################################################
    
    solver = SolverFactory("gurobi") 
        
    solver.solve(model)
    
    assert model.x.extract_values() == {1:2.0,2:-1.0,3:1.0}
    
    assert model.v.extract_values() == {1:-3.0,2:2.0} #{1:3.0,2:-2.0}
    #had to negate the given lambda because N&W Lagrangian assumes a 
    #slightly different form where the lambdas are subtracted    
    
    
def test_Gabriel_quadratic_portfolio_KKT(quadratic_portfolio_Gabriel):
    quadratic_portfolio_Gabriel.initialize_IO_method("Dong_implicit_update") #initialize the method
      
    model = quadratic_portfolio_Gabriel.KKT_conditions_model.clone() 
    
    ## Add constant objective function ##
    model.obj_func = pyo.Objective(expr=5)
    
    #Solve the model
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    for (key,value) in solution.items():
        solution[key] = round(value,2)
    
    assert solution == {1:0.20,2:0.44,3:0.36}
    
def test_Gabriel_quadratic_change_data(quadratic_portfolio_Gabriel):
    quadratic_portfolio_Gabriel.initialize_IO_method("Dong_implicit_update")
    
    ### Modify RHS b ###
    #For this test, all we are going to do is modify the 
    #RHS via receive_data and then we are gonna solve the
    #resulting KKT model to make sure that the RHS got changed correctly
    #Can do this with the other models too!

    quadratic_portfolio_Gabriel.receive_data(p_t={'b':{1:-24}},x_t=np.array([[0],[1],[1]])) 
                                                            #NEW CODE: 4/29/2019 - changed
                                                            #[[0],[1]] to a numpy array
                                                            #ALSO needed to change to 3 part vector
                                                            #WASNT BEING USED IN THE TEST - THAT IS WHY
                                                            #DIDNT GET CAUGHT BEFORE - BUT AN ASSERT
                                                            #STATEMENT WITHIN THE CODE CAUGHT IT
                                                            
                                                            #b wasnt part of the stationary_expression, hence
                                                            #why it ended up working in the test case
    
    
    ###########################################
    model = quadratic_portfolio_Gabriel.KKT_conditions_model.clone()
    
    model.obj_func = pyo.Objective(expr=5)
    
    solver = SolverFactory("gurobi") 
    solver.solve(model)
    
    ## Rounding the Solution ##
    solution = model.x.extract_values()
    for (key,value) in solution.items():
        solution[key] = round(value,2)   
    
    assert solution == {1:0.07,2:0.47,3:0.47} #SHOULD I CHECK MORE THINGS?
    
    
###FOR THE update rule test, we still need to see if we should 
#change the unfixed c values from their current values
    #WE ALSO NEED TO ALLOW USERS TO RESTRICT DOMAIN OF C
    
####FOR the loss function, we can test this using our knowledge of the
##Chan et all paper, bc it is essentially doing a minimum epsilon distance
##Can check it with the Chan example (with a couple different ys)

###We can test the quadratic capabilities and non-negativity capabilities
## (coming soon - do we want an upper bound capability? or should we just make ppl
#in put that as a constraint in the <= ? - Thinking about doing that, but
#do have to solve the bigM problem)
#with the example in Dr. Gabriel's lecture 0 with the 
## portfolio estimation.  We will have to put in dummy data for c
## (just set to 0) because we basically always assume we have a c

##Haven't quite figured out how to test the update rule itself...
    #Can I just do what I was planning to do before and form something
    #on the side?

##Do need to get going on the experiment (if want in time for Jacob
##and want to keep documenting)    
    
###### MORE EXAMPLES #######
    
## REALLY NEED SOMETHING TO TEST OUT THE NEW non_negative thing
## (2) Portfolio stuff   
# (a) WE can test out the changing of the b parameter with this
    #model since we have all of the solutions for the different min returns
    
    
    
