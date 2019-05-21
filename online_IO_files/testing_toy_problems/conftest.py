### conftest.py #####
# 5/18/2019

#According to the pytest documentation, we can create a conftest.py
#file that lets us create ``fixture objects'' (which hold data to pass into
#the unit tests) and then we can import them into multiple test files.

#Source: https://docs.pytest.org/en/latest/fixture.html#fixture

#Therefore, this is our conftest.py file that contains the data for the
#three unit test toy problems that we define in the "Validation" subsection
#of "Overview of Online Methods and the Online_IO Class" section of the Chapter
#Documentation.

#See this subsection for more information.

#We've labelled all of the data constructs underneath their names. 


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
def chan_lee_terekhov_linear_inequalities():
    #Toy Problem 1 Data: Chan, Lee, & Terekhov 2018 Problem (from last semester)
    
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
        
    ### Defining Objective Func ###
    test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(-1),2:(-1)}) #initialize={1:(-2/5),2:(3/5)})
    
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
    ## Toy Problem 2: Quadratic Example from pg 452-453 Nocedal and Wright
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
    
    return test_online

@pytest.fixture
def quadratic_portfolio_Gabriel():
    #Toy Problem 3: Example from Dr. Gabriel's ENME741 and ENME725 Class
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
        
    return test_online
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


