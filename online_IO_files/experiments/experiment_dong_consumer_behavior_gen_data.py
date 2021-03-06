##### experiment_dong_consumer_behavior_gen_data.py #######
# 5/20/2019

# This file produces the data for the "learning consumer preferences" (DCZ 2018)
# computational experiment we carried out from  
# the Dong, Chen, & Zeng 2018 paper.

# Readers can find more information about this data generation process by going to
# Section 1.4.4 Validation/Usage and scrolling to the part that says
# Computational Experiment.

# Note that we are following the data generation process that DCZ outlined in
# their paper.

# Additional Notes:

# DCZ = Dong, Chen, & Zeng

import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import pytest
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import pickle

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing

###### Step -1: Deciding Number of Samples to Generate ######

num_samples = 1000

####### Step 0: Create the Forward Model #######

## Setting up the Index Sets/Basics ##
forward_model = pyo.ConcreteModel()
forward_model.varindex = pyo.RangeSet(1,10)
forward_model.x = pyo.Var(forward_model.varindex,domain=pyo.NonNegativeReals)
forward_model.numvars = pyo.Param(initialize=10)
forward_model.eqindex = pyo.RangeSet(1,1)

## Data ##
forward_model.p_t = pyo.Param(forward_model.eqindex,\
                        forward_model.varindex,initialize=1,mutable=True) #initializing with
                                                        #a dummy value

forward_model.bscalar = pyo.Param(initialize=40)

diag_vec = np.array([2.360,3.465,3.127,0.0791,4.886,2.110,\
        9.519,9.999,2.517,9.867]) 
Q = np.diag(diag_vec) #THIS IS A POSITIVE DEFINITE MATRIX (DCZ wrote a positive def
                        #matrix in their supplemental material when they
                        #meant to write a negative definite)

def Q_param_rule(model,i,j):
    return Q[i-1,j-1]

forward_model.Qmat = pyo.Param(forward_model.varindex,\
                forward_model.varindex,rule=Q_param_rule)

c_dict = {1:-1.180,2:-1.733,3:-1.564,4:-0.040,5:-2.443,6:-1.055,\
          7:-4.760,8:-5.00,9:-1.258,10:-4.933} #made r negative since we will want to minimize

forward_model.cvec = pyo.Param(forward_model.varindex,\
                               initialize=c_dict)

#### Creating the Constraints/Objective Function ####

## Constraints ##
def budget_constraint_rule(model,k):
    return sum(model.p_t[k,i]*model.x[i] \
               for i in range(1,model.numvars+1)) <= model.bscalar

forward_model.budget_constraint = pyo.Constraint(forward_model.eqindex,\
                                            rule=budget_constraint_rule)

## Objective ##
def utility_func(model):
    return 0.5*sum(model.Qmat[i,i]*(model.x[i]**2) for i in range(1,model.numvars+1)) +\
                sum(model.cvec[i]*model.x[i] for i in range(1,model.numvars+1))

forward_model.obj_rule = pyo.Objective(rule=utility_func,sense=pyo.minimize)


################# Step 1: Generate p_t and y_t ###########################
#Pyomo really does work with dictionaries
p_t_samples_dict = {}
y_t_samples_dict = {}

for ns in range(1,num_samples+1):
    p_t = np.random.uniform(5,25,(10,)) #10 random prices
    p_t_dict = {}
    
    for i in range(10):
        p_t_dict[(1,i+1)] = p_t[i] #need the dictionaries to be indexed from 1   
        
    forward_model.p_t.reconstruct(data=p_t_dict) 
    forward_model.budget_constraint.reconstruct() #for now, fixed it by just reconstructing
                                                #the constraint objects individually
    
    #### Solving the Model #####
    solver = SolverFactory("gurobi") 
    solver.solve(forward_model)
    
    print("done with iteration of DCZ data gen") #good for when want to get data
    
    x_t_dict = forward_model.x.extract_values() #get the exact x solution in dictionary form
    y_t_np_array = np.zeros((10,1))
    
    for key,value in x_t_dict.items(): #need to transform into a column vector
        y_t_np_array[key-1,0] = value + np.random.uniform(-0.25,0.25)
        
    ## Putting things in Dictionaries at the End ##
    p_t_samples_dict[ns] = p_t_dict
    y_t_samples_dict[ns] = y_t_np_array
    
    
## Save the Data to Files in the Experiment Folder ##
#Thanks to the following link for helping with this:
    #https://pythonprogramming.net/python-pickle-module-save-objects-serialization/
#We followed steps from this link
write_pickle_file_name = open("dong_p_t.pickle","wb") #think contains file info
pickle.dump(p_t_samples_dict,write_pickle_file_name)
write_pickle_file_name.close()

write_pickle_file_name2 = open("dong_y_t.pickle","wb")
pickle.dump(y_t_samples_dict,write_pickle_file_name2)
write_pickle_file_name2.close()




