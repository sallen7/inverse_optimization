#### Trying out bigM Bound ####

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing

### Example from Nocedal and Wright pg 475 ###
###### Setting Up the Data ######

Qmat = np.array([[1,0],[0,1]]) #Q matrix
Amat = np.array([[-1,2],[1,2],[1,-2],[-1,0],[0,-1]]) #constraints, including non-negative

### Setting Up the Model ###
test_model = pyo.ConcreteModel()
test_model.xvarindex = pyo.RangeSet(2)
#test_model.x = pyo.Var(test_model.varindex)
test_model.inequal_index = pyo.RangeSet(5)

### Importing the G and A Matrices as Parameters ###
#A matrix now contains the parameters for the equality constriants
def A_mat_func(model,i,j):
    return Amat[i-1,j-1]

test_model.Amatrix = pyo.Param(test_model.inequal_index,test_model.xvarindex,rule=A_mat_func)

def Q_mat_func(model,i,j): #which is the Q matrix
    return Qmat[i-1,j-1]

test_model.Qmatrix = pyo.Param(test_model.xvarindex,test_model.xvarindex,rule=Q_mat_func)

### Defining the Vector Parameters ###
#bvector now contains the RHS for the equality constraints
test_model.cvector = pyo.Param(test_model.xvarindex,initialize={1:-2,2:-5})
test_model.bvector = pyo.Param(test_model.inequal_index,initialize={1:2,2:6,3:2,4:0,5:0})

cvector_np = np.array([[-2],[-5]])

### Defining the Constraints ###
test_model.uvarindex = pyo.RangeSet(5)
test_model.u = pyo.Var(test_model.uvarindex)

c_Q_c = np.dot(np.dot(np.transpose(cvector_np),Qmat),cvector_np)

A_Q_c = np.dot(np.dot(Amat,Qmat),cvector_np)

A_Q_A = np.dot(np.dot(Amat,Qmat),np.transpose(Amat))

#pdb.set_trace()

def giant_constraint(model):
    constant = (-1/2)*c_Q_c[0,0]
    A_Q_c_part = (-1/2)*sum(model.u[i]*A_Q_c[i-1,0] for i in range(1,6))
    A_Q_A_part = (-1/2)*sum(sum(A_Q_A[i-1,j-1]*model.u[i]*model.u[j] for j in range(1,6)) for i in range(1,6))
    
    return 1 + (2.5)*(2.5) + constant + 2*A_Q_c_part + A_Q_A_part <= -2

test_model.constraint1 = pyo.Constraint(rule=giant_constraint)

pdb.set_trace()

### Defining the objective function ###
def obj_func(model):
    return sum(model.u[i]**2 for i in range(1,6))

test_model.obj_func = pyo.Objective(rule=obj_func)

solver = SolverFactory("gurobi") #going to go with most high power solver for now
    
results = solver.solve(test_model)
print("This is the termination condition (solve_subproblem):",results.solver.termination_condition)

print("These are the values for u")
for i in range(1,6):
    print('uvalue')
    print(pyo.value(test_model.u[i]))

pdb.set_trace()












#test_online = Online_IO(test_model,Qname='Gmatrix',cname='cvector',\
#        Aname='None',bname='None',Dname='Amatrix',fname='bvector',\
#        dimQ=(3,3),dimc=(3,1),dimD=(2,3),binary_mutable=[0,0,0,0,1,1],non_negative=0)



















