### experiment_BMPS_consumer_behavior_test_Online_GIO.py ###
# 5/20/2019

# This file runs the experiment for the "learning customer preferences" (BMPS 2018)
# computational experiment we carried out (with a few minor adjustments) from  
# the B\"armann, Martin, Pokutta, & Schneider 2018 paper.

# Readers can find more information about running this experiment by going to
# Section 1.3.4 Validation/Usage and scrolling to the part that says
# Computational Experiment.

# We made a few minor adjustments to the experiment BMPS ran. We choose a different
# eta_t and we chose the C set to be a bit different as well.  See the section for details

# Additional Notes:

# Since we had to turn this problem into a minimization problem (since
# Online_IO assumes minimization), we will be finding -u vectors

# Thanks to BMPS for demonstrating the use of a 1/sqrt(t) function to show 
# off the fact that their error graphs were bounded by O(1/sqrt(T))

# BMPS = B\"armann, Martin, Pokutta, & Schneider


import sys
sys.path.insert(0,"C:\\Users\\StephanieAllen\\Documents\\1_AMSC663\\Repository_for_Code")

import copy
import time
import pdb #for debugging
import numpy as np                     
import pyomo.environ as pyo
from pyomo.opt import SolverFactory 
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt #http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html#simple-plot
import pickle

from online_IO_files.online_IO_source.online_IO import Online_IO #importing the GIO class for testing
#from experiment_BMPS_consumer_behavior_gen_data import num_samples #NOTE: if you comment this out
                                                        #then it will actually use the data you already
                                                        #produced when you ran the generating data script
                                                        #ALSO:^don't need to introduce module sequence because we run this script within the
                                                        #experiments directory

#### Step -1: State the amount of data #####
num_samples = 500

##### Step 0: Loading the Data & Set Up Model #####
# Thanks to: 
#https://stackoverflow.com/questions/27570789/do-i-still-need-to-file-close-when-dumping-a-pickle-in-one-line
# for suggesting that we should be closing out the files that we open

#### Load Data ####

loading_data_info = open("BMPS_p_t.pickle","rb")
p_t_samples = pickle.load(loading_data_info) 
loading_data_info.close()

loading_data_info2 = open("BMPS_x_t.pickle","rb")
x_t_samples = pickle.load(loading_data_info2)
loading_data_info2.close()

loading_data_info3 = open("BMPS_RHS_t.pickle","rb")
RHS_t_samples = pickle.load(loading_data_info3)
loading_data_info3.close()

loading_data_info4 = open("BMPS_true_utility_vec.pickle","rb")
true_u_vec = pickle.load(loading_data_info4) 
true_u_vec = -1*true_u_vec #since we are minimizing the negative 
#so the guesses that come back will be negative utility values
loading_data_info4.close()



#####    Set up FORWARD Model    #######
#a. Sets and Vars
cb_model_BMPS = pyo.ConcreteModel()
cb_model_BMPS.varindex = pyo.RangeSet(1,50)
cb_model_BMPS.x = pyo.Var(cb_model_BMPS.varindex,domain=pyo.NonNegativeReals)
cb_model_BMPS.numvars = pyo.Param(initialize=50)
cb_model_BMPS.eqindex = pyo.RangeSet(1,1)

#b. Parameters
cb_model_BMPS.p_t = pyo.Param(cb_model_BMPS.eqindex,\
                        cb_model_BMPS.varindex,initialize=0) 
                                                    
cb_model_BMPS.RHS = pyo.Param([1],initialize={1:0})

cb_model_BMPS.uvec = pyo.Param(cb_model_BMPS.varindex,\
                               initialize=-0.1) #initializing a bit above 0 so that dont have constant objective function issue

##### Set Up Feasible Region Model for us #####
#Users MUST use c for the variable

feasible_c_region = pyo.ConcreteModel()

feasible_c_region.varindex = pyo.RangeSet(1,50)
feasible_c_region.c = pyo.Var(feasible_c_region.varindex)

##### Placing Constraints Upon c #####

def less_than_zero(model,i):
    return model.c[i] <= 0

feasible_c_region.less_than_zero_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                     rule=less_than_zero) 

def greater_than_negative_one(model,i):
    return -1 <= model.c[i]

feasible_c_region.greater_than_negative_one_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                            rule=greater_than_negative_one)

##### Step 1: Create an Instance of the Online_IO Class #####
##### & Initiate the BMPS_online_GD Algorithm #####

time_0 = time.process_time() #actual start of the algorithm (see links in
                        #Section 1.5 Code Attribution for timing links)

online_cb = Online_IO(cb_model_BMPS,Qname='None',cname='uvec',Aname='p_t',\
          bname='RHS',Dname='None',fname='None',dimQ=(0,0),dimc=(50,1),\
          dimA=(1,50),dimD=(0,0),binary_mutable=[0,0,1,1,0,0],non_negative=1,\
          feasible_set_C=feasible_c_region,var_bounds=(0,1))

online_cb.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) 


##### Steps 2-4: Run the Algorithm!  #####

#Need to save some data
c_t_dict_vecs = {}
xbar_dict_vecs = {}

##### Steps 2-4: Run the Algorithm!  #####

#Need to save some data
c_t_dict_vecs = {}
xbar_dict_vecs = {}

for i in range(1,num_samples+1):
    ### Step 2: "Project onto C" (from BMPS 2018) ###
    
    online_cb.next_iteration(part_for_BMPS=1) #we have to break the "next iteration"
        #of the BMPS_online_GD algorithm into two parts, since "project onto F"
        #(from BMPS 2018) comes before the data update step
        
    ### Step 3: Update Subproblem with p_t and obtain "expert solution x_t" (from BMPS 2018) ###
    #For p_t, we pass in a dictionary with keys as the names of the parameters 
    #that we are updating and the items attached to the keys as the dictionaries
    #containing the data.
    
    online_cb.receive_data(p_t={"p_t":p_t_samples[i],"RHS":RHS_t_samples[i]},x_t=x_t_samples[i])
    
    ## The next few statements do some basic checks.  They aren't comprehensive, but ##
    ## they are useful to showcase because they indicate some approaches to debugging pyomo models ##
    ## See documentation for sources regarding the pyomo package and its documentation (and some sites I found helpful) ##
    
    # Asserting that the parameters p_t and RHS_t both gets put into the BMPS_subproblem #
    assert online_cb.BMPS_subproblem.A.extract_values() == p_t_samples[i]
    assert online_cb.BMPS_subproblem.b.extract_values() == RHS_t_samples[i]
    
    # Doing Some constraint Checking #
    for constr in online_cb.BMPS_subproblem.component_objects(pyo.Constraint):
        assert constr._constructed == True, "Error in constraint construction (body)"
        for c in constr:
            lb = pyo.value(constr[c].lower)
            ub = pyo.value(constr[c].upper)
            assert ((lb is not None) or (ub is not None)), "Error in constraint construction (LHS/RHS)"
    
    ### Step 4: Finish out the Iteration by "solving subproblem", "performing gradient ###
    ### descent step", and calculating the learning rate (BMPS, 2018)
    
    online_cb.next_iteration(part_for_BMPS=2)
    
    ## Save some data ##
    c_t_dict_vecs[i] = copy.deepcopy(online_cb.c_t_BMPS)
    xbar_dict_vecs[i] = copy.deepcopy(online_cb.xbar_t_BMPS)

    
print("Run time (seconds) for the script: ")             
time_1 = time.process_time() - time_0
print(time_1)      


################ Step 4: Graph BMPS 2018 Error Measures ##################
##### Step 4: Graph BMPS 2018 Error Measures #####
true_u_vec = np.reshape(true_u_vec,(50,1))

BMPS_obj_error = np.zeros((num_samples,))
BMPS_sol_error = np.zeros((num_samples,))
BMPS_total_error = np.zeros((num_samples,))

for i in range(1,num_samples+1): #cannot directly compare graph to paper bc we use 1/sqrt(t) - might need to run for more iterations?
    BMPS_obj_error[i-1] = np.dot(np.transpose(c_t_dict_vecs[i]),(x_t_samples[i]-xbar_dict_vecs[i]))
    BMPS_sol_error[i-1] = np.dot(np.transpose(true_u_vec),(x_t_samples[i]-xbar_dict_vecs[i])) #since we made u_vec as like negative ok?
    BMPS_total_error[i-1] = BMPS_obj_error[i-1] - BMPS_sol_error[i-1] 
 
BMPS_obj_error = np.divide(np.cumsum(BMPS_obj_error),np.arange(1,num_samples+1))
BMPS_sol_error = np.divide(np.cumsum(BMPS_sol_error),np.arange(1,num_samples+1))
BMPS_total_error = np.divide(np.cumsum(BMPS_total_error),np.arange(1,num_samples+1))
    
##############  Creating the Graph  ###################    
plt.subplot(131)
error_graph1 = plt.plot(np.arange(1,num_samples+1),BMPS_obj_error)
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('Objective error: u_t^T (xbar_t-x_t)',fontsize=20)
plt.title('BMPS Objective Error',fontsize=20)


plt.subplot(132)
error_graph2 = plt.plot(np.arange(1,num_samples+1),BMPS_sol_error)
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('Solution error: u_true^T (xbar_t-x_t)',fontsize=20)
plt.title('BMPS Solution Error',fontsize=20)


plt.subplot(133)
constant = 8
bound_func = lambda x: constant*(1/(np.sqrt(x))) #following the lead of BMPS 2018
error_graph3 = plt.plot(np.arange(1,num_samples+1),BMPS_total_error,label="sol_error")
error_graph3 = plt.plot(np.arange(1,num_samples+1),bound_func(np.arange(1,num_samples+1)),'--',\
                        label="regret bound = constant*(1/sqrt(t))")
plt.legend()
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('Total error',fontsize=20)
plt.title('BMPS Total Error',fontsize=20)

## Show the three panel graph ##
plt.show()


###### Total Error Logged ########
error_graph4 = plt.semilogy(np.arange(1,num_samples+1),BMPS_total_error,label="sol_error")
error_graph4 = plt.semilogy(np.arange(1,num_samples+1),bound_func(np.arange(1,num_samples+1)),'--',\
                            label="regret bound = log(constant/sqrt(t))")
plt.legend()
plt.xlabel('Iterations (t)',fontsize=20)
plt.ylabel('Total error (logged)',fontsize=20)
plt.title('BMPS Total Error (Logged on Y axis)',fontsize=20)
plt.show()
