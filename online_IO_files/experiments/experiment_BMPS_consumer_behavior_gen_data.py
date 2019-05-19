### Experiment for B\"armann, Martin, Pokutta, & Schneider 2018 ###

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

##u_i are positive
##T = 500 observations with n = 50 goods for each random instance

num_samples = 500

#### Step 0: Create the Forward Model ####

## Setting up the Index Sets/Basics ##
forward_model_BMPS = pyo.ConcreteModel()
forward_model_BMPS.varindex = pyo.RangeSet(1,50)
forward_model_BMPS.x = pyo.Var(forward_model_BMPS.varindex,bounds=(0,1)) #NEED THE BOUNDS HERE!!!
forward_model_BMPS.numvars = pyo.Param(initialize=50)
forward_model_BMPS.ineqindex = pyo.RangeSet(1,1)

## Data ##
forward_model_BMPS.p_t = pyo.Param(forward_model_BMPS.ineqindex,\
                        forward_model_BMPS.varindex,initialize=1,mutable=True) #initializing with
                                                        #a dummy value

forward_model_BMPS.RHSscalar = pyo.Param(forward_model_BMPS.ineqindex,initialize=40,mutable=True) #NEW THING: This has to be mutable too
                                            #DOES NEED TO BE INDEXED?

utilities = np.random.randint(1,1001,(50,))
utilities = utilities*(1/np.linalg.norm(utilities,ord=1))
##Need to transform the np.array into a dictionary###
#Thanks to: https://stackoverflow.com/questions/31575675/how-to-convert-numpy-array-to-python-dictionary-with-sequential-keys
#Stuff on Enumerate: https://www.geeksforgeeks.org/enumerate-in-python/
#https://www.afternerd.com/blog/python-enumerate/#enumerate-tuple
utilities_dict = dict(enumerate(utilities,1))


forward_model_BMPS.uvec = pyo.Param(forward_model_BMPS.varindex,\
                               initialize=utilities_dict)

## Creating the Constraints/Objective Function ##

## Constraints ##
def budget_constraint_rule(model,k):
    return sum(model.p_t[k,i]*model.x[i] \
               for i in range(1,model.numvars+1)) <= model.RHSscalar[k]

forward_model_BMPS.budget_constraint = pyo.Constraint(forward_model_BMPS.ineqindex,\
                                            rule=budget_constraint_rule)

## Objective ##
def utility_func(model):
    return -1*sum(model.uvec[i]*model.x[i] for i in range(1,model.numvars+1)) #NEED TO NEGATE SINCE MINIMIZING

forward_model_BMPS.obj_rule = pyo.Objective(rule=utility_func,sense=pyo.minimize)


#### Step 1: Generate p_t and y_t ####
#Pyomo really does work with dictionaries
p_t_samples_dict = {}
RHS_t_samples_dict = {}
x_t_samples_dict = {}

for ns in range(1,num_samples+1):
    #### Generating the Prices for Round t ####
    p_t = utilities + 100 + np.random.randint(-10,11,(50,)) #50 semi-random prices
    #p_t_dict = dict(enumerate(p_t,1))   
    p_t_dict = {}
    
    for i in range(50):
        p_t_dict[(1,i+1)] = p_t[i] #need the dictionaries to be indexed from 1 (for the second index)
                                #and have the first index be a 1 because we have ONE inequality
    
    ### Generating RHS for Round t ###
    #NEED TO SEE if this ends up being an integer afterall!
    RHS_t = np.random.randint(1,np.sum(p_t),(1,))
    RHS_t_dict = dict(enumerate(RHS_t,1))   

    #######################################################################
    #### Updating the Model ####    
    forward_model_BMPS.p_t.reconstruct(data=p_t_dict) 
    forward_model_BMPS.budget_constraint.reconstruct() #for now, fixed it by just reconstructing
                                                #the constraint objects individually
    
    #pdb.set_trace() - seems to be good
    forward_model_BMPS.RHSscalar.reconstruct(data=RHS_t_dict)
    forward_model_BMPS.budget_constraint.reconstruct()
    #pdb.set_trace() - seems to be good
    
    ## Solving the Model ##
    solver = SolverFactory("gurobi")
    solver.solve(forward_model_BMPS)
    
    x_t_dict = forward_model_BMPS.x.extract_values() #get the exact x solution in dictionary form
    ### Need the x_t to be an array ###
    x_t_array = np.fromiter(x_t_dict.values(),dtype=float,count=len(x_t_dict))
    x_t_array_col = np.reshape(x_t_array,(len(x_t_dict),1))
    
    print("another BMPS data generation iteration complete")       
        
    #### Putting things in Dictionaries at the End ####
    p_t_samples_dict[ns] = p_t_dict
    x_t_samples_dict[ns] = x_t_array_col
    RHS_t_samples_dict[ns] = RHS_t_dict
    
## Save the Data to Files in the Experiment Folder ##
#Thanks to the following link for helping with this:
    #https://pythonprogramming.net/python-pickle-module-save-objects-serialization/
#We followed steps from this link
pickle_out = open("BMPS_p_t.pickle","wb") #think contains file info
pickle.dump(p_t_samples_dict,pickle_out)
pickle_out.close()

pickle_out = open("BMPS_x_t.pickle","wb")
pickle.dump(x_t_samples_dict,pickle_out)
pickle_out.close()

pickle_out = open("BMPS_RHS_t.pickle","wb")
pickle.dump(RHS_t_samples_dict,pickle_out)
pickle_out.close()

pickle_out = open("BMPS_true_utility_vec.pickle","wb")
pickle.dump(utilities,pickle_out)
pickle_out.close()






