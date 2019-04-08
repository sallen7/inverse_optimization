#### Working with the Chan et al. Example ####

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")


import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.expr import current as EXPR

#print(sys.path)
#pdb.set_trace()

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing


##### Defining the Chan et al. (2018) Model #####
# Chan et al. assumed Ax >= b
# We are (for the sake of the KKT conditions) assuming Ax <= b

A = np.array([[2,5],[2,-3],[2,1],[-2,-1]])
#b = np.array([[10],[-6],[4],[-10]])
x0 = np.array([[2.5],[3]])
        
test_model = pyo.ConcreteModel()
test_model.varindex = pyo.RangeSet(2)
test_model.numvars = pyo.Param(initialize=2)
test_model.eqindex = pyo.RangeSet(4)
test_model.numeqs = pyo.Param(initialize=4)
test_model.x = pyo.Var(test_model.varindex)

#pdb.set_trace()

##### Importing the A and b as param objects #####
def A_mat_func(model,i,j):
    return (-1)*A[i-1,j-1]

test_model.Amat = pyo.Param(test_model.eqindex,test_model.varindex,rule=A_mat_func)
test_model.bvec = pyo.Param(test_model.eqindex,initialize={1:-10,2:6,3:-4,4:10})

def constraint_rule(model,i):
    return sum(model.Amat[i,j]*model.x[j] for j in range(1,3)) <= model.bvec[i]

test_model.Abconstraints = pyo.Constraint(test_model.eqindex,rule=constraint_rule)

### Defining Objective Func ###
#MIGHT have an issue with the SIGNS of the c parameters
test_model.cvec = pyo.Param(test_model.varindex,initialize={1:(2/5),2:(-3/5)}) #initialize={1:(-2/5),2:(3/5)})

def obj_rule_func(model):
    return model.cvec[1]*model.x[1]+model.cvec[2]*model.x[2]

test_model.obj_func = pyo.Objective(rule=obj_rule_func)


test_model.dummy_param = pyo.Param()

#pdb.set_trace()


#### Creating an Instance of the Class with Chan et al. Model ####
#def __init__(self,initial_model,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
#                 dimQ=(1,1),dimc=(1,1),dimA=(1,1),dimD=(1,1)):

#TO DO: See if the dummy parameter actually works, and then if it does
#thinking about a work around

#def __init__(self,initial_model,Qname='Q',cname='c',Aname='A',bname='b',Dname='D',fname='f',\
#                 dimQ=(1,1),dimc=(1,1),dimA=(1,1),dimD=(1,1),binary_mutable=[0,0,0,0,0,0]):

chan_online = Online_IO(test_model,Qname='None',cname='cvec',Aname='Amat',\
                        bname='bvec',Dname='None',fname='None',\
                        dimQ=(0,0),dimc=(2,1),dimA=(4,2),dimD=(0,0),binary_mutable=[0,0,0,1,0,0])

#sample_y = np.array([[2.5],[3]]) #maybe the model is all pyomo but the 
                                #input data is numpy arrays
chan_online.initialize_IO_method("Dong_implicit_update")
#chan_online.loss_function(y=sample_y)


#chan_online.loss_model_dong.pprint() #see the solution
                            #DID indeed get the right values once I change
                            # the objective function c back to the right set of values
                            
                            #Looks like the x is being set right
                            #WHY IS THIS A_index set created??

##### Feeding in New Data into the Model #####

test_dict = {}
new_b = {1:100,2:100,3:100,4:100} 
test_dict['bvec'] = new_b

chan_online.receive_data(p_t=test_dict)

chan_online.KKT_conditions_model.pprint()

pdb.set_trace() #it worked!!! - no it didn't, new_b was updated but the constraints werent
#you have to call a "reconstruct" on the constraint if you want it to fully update
                            
                            
                















