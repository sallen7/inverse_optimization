### experiment_BMPS_consumer_behavior_gen_data.py ###
# 5/20/2019

# This file produces the data for the "learning customer preferences" (BMPS 2018)
# computational experiment we carried out (with a few minor adjustments) from  
# the B\"armann, Martin, Pokutta, & Schneider 2018 paper.

# Readers can find more information about this data generation process by going to
# Section 1.3.4 Validation/Usage and scrolling to the part that says
# Computational Experiment.

# Note that we are following the data generation process that BMPS outlined in
# their paper

# Additional notes:

# BMPS = B\"armann, Martin, Pokutta, & Schneider

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


############### Step -1: Define the amount of data that we need to generate ##################
num_samples = 500

############## Step 0: Create the Forward Model #####################

## Setting up the Index Sets/Basics ##
forward_model_BMPS = pyo.ConcreteModel()
forward_model_BMPS.varindex = pyo.RangeSet(1,50)
forward_model_BMPS.x = pyo.Var(forward_model_BMPS.varindex,bounds=(0,1)) 
forward_model_BMPS.numvars = pyo.Param(initialize=50)
forward_model_BMPS.ineqindex = pyo.RangeSet(1,1)

## Data ##
forward_model_BMPS.p_t = pyo.Param(forward_model_BMPS.ineqindex,\
                        forward_model_BMPS.varindex,initialize=1,mutable=True) #initializing with
                                                        #a dummy value

forward_model_BMPS.RHSscalar = pyo.Param(forward_model_BMPS.ineqindex,initialize=40,mutable=True) #NEW THING: This has to be mutable too

##### Randomly generating the utilities vector ######
utilities = np.random.randint(1,1001,(50,))
utilities = utilities*(1/np.linalg.norm(utilities,ord=1))

##Need to transform the np.array into a dictionary###
#Thanks to: https://stackoverflow.com/questions/31575675/how-to-convert-numpy-array-to-python-dictionary-with-sequential-keys
#Stuff on Enumerate: https://www.geeksforgeeks.org/enumerate-in-python/
#https://www.afternerd.com/blog/python-enumerate/#enumerate-tuple
utilities_dict = dict(enumerate(utilities,1))


forward_model_BMPS.uvec = pyo.Param(forward_model_BMPS.varindex,\
                               initialize=utilities_dict)

##### Creating the Constraints/Objective Function ######

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


################# Step 1: Generate p_t and x_t ################################
#Pyomo really does work with dictionaries
p_t_samples_dict = {}
RHS_t_samples_dict = {}
x_t_samples_dict = {}

for ns in range(1,num_samples+1):
    #### Generating the Prices for Round t ####
    p_t = utilities + 100 + np.random.randint(-10,11,(50,)) #50 semi-random prices   
    p_t_dict = {}
    
    for i in range(50):
        p_t_dict[(1,i+1)] = p_t[i] #need the dictionaries to be indexed from 1 (for the second index)
                                #and have the first index be a 1 because we have ONE inequality
    
    ### Generating RHS for Round t ###
    RHS_t = np.random.randint(1,np.sum(p_t),(1,))
    RHS_t_dict = dict(enumerate(RHS_t,1))   

    #### Updating the Model ####    
    forward_model_BMPS.p_t.reconstruct(data=p_t_dict) 
    forward_model_BMPS.budget_constraint.reconstruct() #for now, fixed it by just reconstructing
                                                #the constraint objects individually
    
    forward_model_BMPS.RHSscalar.reconstruct(data=RHS_t_dict)
    forward_model_BMPS.budget_constraint.reconstruct()
    
    ##### Solving the Model #####
    solver = SolverFactory("gurobi")
    solver.solve(forward_model_BMPS)
    
    x_t_dict = forward_model_BMPS.x.extract_values() #get the exact x solution in dictionary form
    x_t_array = np.fromiter(x_t_dict.values(),dtype=float,count=len(x_t_dict)) 
    x_t_array_col = np.reshape(x_t_array,(len(x_t_dict),1)) # need the x_t to be an array
    
    print("another BMPS data generation iteration complete") #helps to know this when we 
                                                            #actually generate the data      
        
    #### Putting data in Dictionaries at the End ####
    p_t_samples_dict[ns] = p_t_dict
    x_t_samples_dict[ns] = x_t_array_col
    RHS_t_samples_dict[ns] = RHS_t_dict
    
## Save the Data to Files in the Experiment Folder ##
#Thanks to the following link for helping with this:
    #https://pythonprogramming.net/python-pickle-module-save-objects-serialization/
#We followed steps from this link
write_pickle_file_name = open("BMPS_p_t.pickle","wb") #think contains file info
pickle.dump(p_t_samples_dict,write_pickle_file_name)
write_pickle_file_name.close()

write_pickle_file_name2 = open("BMPS_x_t.pickle","wb")
pickle.dump(x_t_samples_dict,write_pickle_file_name2)
write_pickle_file_name2.close()

write_pickle_file_name3 = open("BMPS_RHS_t.pickle","wb")
pickle.dump(RHS_t_samples_dict,write_pickle_file_name3)
write_pickle_file_name3.close()

write_pickle_file_name4 = open("BMPS_true_utility_vec.pickle","wb")
pickle.dump(utilities,write_pickle_file_name4)
write_pickle_file_name4.close()






