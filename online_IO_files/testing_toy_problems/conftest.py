####### Setting Up My conftest.py File for the Toy Test Problems ##########

#According to the pytest documentation, we can create this conftest.py
#file that lets us create ``fixture objects'' (which hold data to pass into
#the unit tests) and then we can import them into multiple test files.

#With this script, I think we can get at the whole "no dependency" thing
#that the Hitchhikers Guide wanted for unit testing.  I can make 
#several fixtures of the same model (ie Chan et al) that are at different
#stages of the process, which will allow me to unit test everything 
#separately. Ex: will want an initialization that doesnt call the
#initialize_IO_method and tries to do the other two (will want two separate
#things to test the two separate method

#Might be able to also specify some separate data stuff to test the Dong 
#specific functions

#According to the documentation, we can also
#(1) Deal with when "the result of a fixture 
#is needed multiple times in a single test"
#See the factories as fixtures section
#(2) The fixtures themselves can take as input
#a "params" argument which  is "a list of values
#for each of which the fixture function will 
#execute".  There is a little bit more mechanics
#to making this work
#See Parameterizing fixtures section

#Source: https://docs.pytest.org/en/latest/fixture.html#fixture


import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing

@pytest.fixture
def chan_init_IO_method():
    #Fixture 2: Checking the initialize_IO_method
    #method from the Online_IO class
    
    A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    x0 = np.array([[2.5],[3]])
            
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(2)
    test_model.numvars = pyo.Param(initialize=2)
    test_model.eqindex = pyo.RangeSet(4)
    test_model.numeqs = pyo.Param(initialize=4)
    
    ##### Importing the A and b as param objects #####
    def A_mat_func(model,i,j):
        return (-1)*A[i-1,j-1]
    
    test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})
        
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(2/5),2:(-3/5)})    
    
    ###############################################
    test_online = Online_IO(test_model,Qname='None',cname='cvec',\
            Aname='Amat',bname='bvec',Dname='None',fname='None',\
            dimc=(2,1),dimA=(4,2),binary_mutable=[0,0,0,1,0,0])
    
    test_online.initialize_IO_method("Dong_implicit_update")
    
    return test_online 

@pytest.fixture
def chan_compute_KKT_1():
    #Fixture 3: Checking the compute_KKT_conditions method
    #with Chan et al example
    
    A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    #x0 = np.array([[2.5],[3]])
            
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(2)
    test_model.numvars = pyo.Param(initialize=2)
    test_model.eqindex = pyo.RangeSet(4)
    test_model.numeqs = pyo.Param(initialize=4)
    
    ##### Importing the A and b as param objects #####
    def A_mat_func(model,i,j):
        return (-1)*A[i-1,j-1]
    
    test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})
        
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:-1,2:-1})    
    
    ###############################################
    test_online = Online_IO(test_model,Qname='None',cname='cvec',\
            Aname='Amat',bname='bvec',Dname='None',fname='None',\
            dimc=(2,1),dimA=(4,2),binary_mutable=[0,0,0,1,0,0])
    
    test_online.initialize_IO_method("Dong_implicit_update")
    
    ### Adding Objective Function Onto test_online.KKT
    test_online.KKT_conditions_model.obj_func = pyo.Objective(expr=5) #feasible point is optimal
                                                    #because of linear programming theory
                                                    
    return test_online

@pytest.fixture
def chan_compute_KKT_2():
    #Fixture 3: Checking the compute_KKT_conditions method
    #with Chan et al example
    
    A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    #x0 = np.array([[2.5],[3]])
            
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(2)
    test_model.numvars = pyo.Param(initialize=2)
    test_model.eqindex = pyo.RangeSet(4)
    test_model.numeqs = pyo.Param(initialize=4)
    
    ##### Importing the A and b as param objects #####
    def A_mat_func(model,i,j):
        return (-1)*A[i-1,j-1]
    
    test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})
        
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(3/5),2:(-2/5)})    
    
    ###############################################
    test_online = Online_IO(test_model,Qname='None',cname='cvec',\
            Aname='Amat',bname='bvec',Dname='None',fname='None',\
            dimc=(2,1),dimA=(4,2),binary_mutable=[0,0,0,1,0,0])
    
    test_online.initialize_IO_method("Dong_implicit_update")
    
    ### Adding Objective Function Onto test_online.KKT
    test_online.KKT_conditions_model.obj_func = pyo.Objective(expr=5) #feasible point is optimal
                                                    #because of linear programming theory
                                                    
    return test_online  


@pytest.fixture
def chan_receive_data():
    #Fixture 2: Checking the initialize_IO_method
    #method from the Online_IO class
    
    A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    x0 = np.array([[2.5],[3]])
            
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(2)
    test_model.numvars = pyo.Param(initialize=2)
    test_model.eqindex = pyo.RangeSet(4)
    test_model.numeqs = pyo.Param(initialize=4)
    
    ##### Importing the A and b as param objects #####
    def A_mat_func(model,i,j):
        return (-1)*A[i-1,j-1]
    
    test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})
        
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(2/5),2:(-3/5)})    
    
    ###############################################
    test_online = Online_IO(test_model,Qname='None',cname='cvec',\
            Aname='Amat',bname='bvec',Dname='None',fname='None',\
            dimc=(2,1),dimA=(4,2),binary_mutable=[0,0,0,1,0,0])
    
    test_online.initialize_IO_method("Dong_implicit_update")
    
    ### Throwing in Data when Initialize ###
    #NOTE: We actually have the ability for the 
    #method to not HAVE to receive data 
    #(because hey you may not for an iteration)
    #by leaving the default parameters alone
    
    test_online.receive_data(x_t = x0)
    
    return test_online 

############################################################################
    #Why are there 4 chan things above?
    #Probably good to have the two chans as separate fixtures (with the two
    #different cvector values) 
    
@pytest.fixture
def chan_lee_terekhov_linear_inequalities():
    #Fixture 1: Checking Initialization of Online_IO
    #instance
    
    #Pretty sure we don't need to define the entire
    #model anyway
    A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
    x0 = np.array([[2.5],[3]])
            
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(2)
    test_model.numvars = pyo.Param(initialize=2)
    test_model.eqindex = pyo.RangeSet(4)
    test_model.numeqs = pyo.Param(initialize=4)
    test_model.x = pyo.Var(test_model.varindex)
    
    ##### Importing the A and b as param objects #####
    def A_mat_func(model,i,j):
        return (-1)*A[i-1,j-1]
    
    test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})
    
    def constraint_rule(model,i):
        return sum(model.Amat[i,j]*model.x[j] for j in range(1,3)) <= model.bvec[i]
    
    test_model.Abconstraints = pyo.Constraint(test_model.eqindex,rule=constraint_rule)
    
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(-1),2:(-1)}) #initialize={1:(-2/5),2:(3/5)})
    
    def obj_rule_func(model):
        return model.cvec[1]*model.x[1]+model.cvec[2]*model.x[2]
    
    test_model.obj_func = pyo.Objective(rule=obj_rule_func)
    
    
    ###############################################
    ##### Finally Calling the Online_IO Class #####
    
    test_online = Online_IO(test_model,Qname='None',cname='cvec',\
            Aname='Amat',bname='bvec',Dname='None',fname='None',\
            dimc=(2,1),dimA=(4,2),binary_mutable=[0,0,1,1,0,0])
    #NOTE: We did not fill in all dimensions, just the 
    #ones where the defaults were not true
    
    return test_online

@pytest.fixture
def quadratic_NW():
    ## Quadratic Example from pg 452-453 Nocedal and Wright
    #Has quadratic objective, equality constraints
    #The Q matrix is symmetric positive definite.
    
    Gmat = np.array([[6,2,1],[2,5,2],[1,2,4]]) #actually a Q matrix
    Amat = np.array([[1,0,1],[0,1,1]]) #actually a D matrix
    
    ### Setting Up the Model ###
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(3)
    test_model.numvars = pyo.Param(initialize=3)
    test_model.eqindex = pyo.RangeSet(2)
    test_model.numeqs = pyo.Param(initialize=2)
    
    ### Importing the G and A Matrices as Parameters ###
    #A matrix now contains the parameters for the equality constriants
    def A_mat_func(model,i,j):
        return Amat[i-1,j-1]
    
    test_model.Amatrix = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
    
    def G_mat_func(model,i,j): #which is the Q matrix
        return Gmat[i-1,j-1]
    
    test_model.Gmatrix = pyo.Param(test_model.varindex,test_model.varindex,rule=G_mat_func)
    
    ### Defining the Vector Parameters ###
    #bvector now contains the RHS for the equality constraints
    test_model.cvector = pyo.Param(test_model.varindex,initialize={1:-8,2:-3,3:-3})
    test_model.bvector = pyo.Param(test_model.eqindex,initialize={1:3,2:0})
    
    
    test_online = Online_IO(test_model,Qname='Gmatrix',cname='cvector',\
            Aname='None',bname='None',Dname='Amatrix',fname='bvector',\
            dimQ=(3,3),dimc=(3,1),dimD=(2,3),binary_mutable=[0,0,0,0,1,1],non_negative=0)
    #NEW CODE 4/29/2019 - Changing [0,0,0,0,0,1] to [0,0,0,0,1,1]
    
    
    #test_online.initialize_IO_method("Dong_implicit_update") #NEW THING: 5/2/2019
    
    ### Adding Objective Function: As long as Satisfy KKT N&S then
    ## Can use Constant Objective Function 
    #test_online.KKT_conditions_model.obj_func = pyo.Objective(expr=5)
    #NEW CODE 4/29/2019 - Commenting out this adding of the objective function
    #since I want to use this set up of the data for multiple tests in the
    #test_DCZ_code script
    
    return test_online

@pytest.fixture
def quadratic_portfolio_Gabriel():
    #Example from Dr. Gabriel's ENME741 and ENME725 Class
    #Variation on the Classical Portfolio Example
    
    ### Data ###
    V = np.array([[4.2,-1.9,2],[-1.9,6.7,-5],[2,-5,7]])    
    
    ### Setting Up the Model ###
    test_model = pyo.ConcreteModel()
    test_model.varindex = pyo.RangeSet(3)
    test_model.numvars = pyo.Param(initialize=3)
    test_model.eqindex = pyo.RangeSet(1)
    test_model.numeq = pyo.Param(initialize=1)
    
    ### Inputting Params ###
    #Need to remember that the inequalities are Ax <= b (so flip things
    #as needed)
    #Need to also make clear what assumptions you are making about the structure
    #of your data.
    test_model.DMaTrix = pyo.Param(test_model.eqindex,\
        test_model.varindex,initialize={(1,1):1,(1,2):1,(1,3):1})
    test_model.f = pyo.Param([1],initialize={1:1})
    test_model.AmAtRix = pyo.Param(test_model.eqindex,\
        test_model.varindex,initialize={(1,1):-10,(1,2):-20,(1,3):-30})
    test_model.b = pyo.Param([1],initialize={1:-20})
    
    def V_mat_func(model,i,j): #which is the Q matrix
        return V[i-1,j-1]
    
    test_model.Vmat = pyo.Param(test_model.varindex,test_model.varindex,\
                                rule=V_mat_func)
    
    #Need a dummy c because it is basically the only parameter in which
    #we absolutely assume we have
    test_model.cdummy = pyo.Param(test_model.varindex,initialize={1:0,2:0,3:0})
    
    #REMEMBER: this is the model in which we SPECIFY non-negative variables
    test_online = Online_IO(test_model,Qname='Vmat',cname='cdummy',\
            Aname='AmAtRix',bname='b',Dname='DMaTrix',fname='f',\
            dimQ=(3,3),dimc=(3,1),dimA=(1,3),dimD=(1,3),\
            binary_mutable=[0,0,0,1,0,0],non_negative=1)
    
    
    #test_online.initialize_IO_method("Dong_implicit_update") #NEW THING 5/2/2019 - commented out
    
    ### Adding Objective Function: As long as Satisfy KKT N&S then
    ## Can use Constant Objective Function 
    #test_online.KKT_conditions_model.obj_func = pyo.Objective(expr=5) #NEW THING 5/2/2019 - commented out
    
    return test_online
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


