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
from experiment_BMPS_consumer_behavior_gen_data import num_samples
#^don't need to introduce module sequence because we run this script within the
#experiments directory

#REMEMBER, THIS IS A MINIMIZATION PROBLEM, SO YOU NEED TO adjust everything
#so that the us are NEGATIVE!!!!!


##### Step 0: Loading the Data & Set Up Model #####
###    Load Data    ###
# Thanks to: 
#https://stackoverflow.com/questions/27570789/do-i-still-need-to-file-close-when-dumping-a-pickle-in-one-line
# for suggesting that we should be closing out the files that we open

loading_data_info = open("BMPS_p_t.pickle","rb")
p_t_samples = pickle.load(loading_data_info) 
loading_data_info.close()

#pdb.set_trace()

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

#pdb.set_trace()

#####    Set up FORWARD Model    #######
#a. Sets and Vars
cb_model_BMPS = pyo.ConcreteModel()
cb_model_BMPS.varindex = pyo.RangeSet(1,50)
cb_model_BMPS.x = pyo.Var(cb_model_BMPS.varindex,domain=pyo.NonNegativeReals)
cb_model_BMPS.numvars = pyo.Param(initialize=50)
cb_model_BMPS.eqindex = pyo.RangeSet(1,1)

#b. Parameters
cb_model_BMPS.p_t = pyo.Param(cb_model_BMPS.eqindex,\
                        cb_model_BMPS.varindex,initialize=0,mutable=True) 
                                                    
cb_model_BMPS.RHS = pyo.Param([1],initialize={1:40})

cb_model_BMPS.uvec = pyo.Param(cb_model_BMPS.varindex,\
                               initialize=-0.1) #INITIALIZING a bit above 0 so that dont have constant objective function issue

##### Set Up Feasible Region Model for us #####
#I do have to use the "c" variable in my functions though

feasible_c_region = pyo.ConcreteModel()

feasible_c_region.varindex = pyo.RangeSet(1,50)
feasible_c_region.c = pyo.Var(feasible_c_region.varindex)

#pdb.set_trace()

##### Placing Constraints Upon c #####
#We need all constraints to be in the form of rules (including
#if we have non-negativity constraints)

#Will need to see if _index works

def less_than_zero(model,i):
    return model.c[i] <= 0

feasible_c_region.less_than_zero_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                     rule=less_than_zero) 

def greater_than_negative_one(model,i):
    return -1 <= model.c[i]

feasible_c_region.greater_than_negative_one_constraint = pyo.Constraint(feasible_c_region.varindex,\
                                                            rule=greater_than_negative_one)

#def sum_to_negative_1(model):
#    return sum(model.c[j] for j in range(1,51)) == -1
#
#feasible_c_region.sum_to_negative_1_constraint = pyo.Constraint(rule=sum_to_negative_1)


##### Step 1: Create an Instance of the Online_IO Class #####
##### & Initiate the BMPS_online_GD Algorithm #####

#Note that initial guess of c is vector of 1s

time_0 = time.process_time() #actual start of the algorithm

#pdb.set_trace()

online_cb = Online_IO(cb_model_BMPS,Qname='None',cname='uvec',Aname='p_t',\
          bname='RHS',Dname='None',fname='None',dimQ=(0,0),dimc=(50,1),\
          dimA=(1,50),dimD=(0,0),binary_mutable=[0,0,1,1,0,0],non_negative=1,\
          feasible_set_C=feasible_c_region,var_bounds=(0,1))


#NOTE: TO RUN THIS EXPERIMENT NOW, I NEED TO THROW IN CONSTRAINTS INVOLVING keeping the x BELOW 1
#would require me to generate a completely different p_t - MUCH bigger
#might need to actually reintroduce var_bounds for this experiment to work easily...
#GOOD THING: we do KNOW that it works - just need to get stuff cleaned up

#Q,c,A,b,D,f

online_cb.initialize_IO_method("BMPS_online_GD",alg_specific_params={'diam_flag':0}) #remember we assume the first y_1
            #vector is composed of the c values that the model starts with


##### Steps 2-4: Run the Algorithm!  #####

#Need to save some data
c_t_dict_vecs = {}
xbar_dict_vecs = {}


for i in range(1,num_samples+1):
    ### Step 2: "Project onto F" (from BMPS 2018) ###
    online_cb.next_iteration(part_for_BMPS=1) #we have to break the "next iteration"
        #of the BMPS_online_GD algorithm into two parts, since "project onto F"
        #(from BMPS 2018) comes before the data update step
        
    ### Step 3: Update Subproblem with p_t and obtain "expert solution x_t" (from BMPS 2018) ###
    #For p_t, we pass in a dictionary with keys as the names of the parameters 
    #that we are updating and the items attached to the keys as the dictionaries
    #containing the data.
    online_cb.receive_data(p_t={"p_t":p_t_samples[i],"RHS":RHS_t_samples[i]},x_t=x_t_samples[i])
    
    # Asserting that the parameters p_t and RHS_t both gets put into the BMPS_subproblem #
    assert online_cb.BMPS_subproblem.A.extract_values() == p_t_samples[i]
    assert online_cb.BMPS_subproblem.b.extract_values() == RHS_t_samples[i]
    
    # Making sure all the constraints were updated (and at least one side is not None) #
    for constr in online_cb.BMPS_subproblem.component_objects(pyo.Constraint):
        assert constr._constructed == True, "Error in constraint construction (body)"
        for c in constr:
            #assert constr[c]._constructed == True, "Error in constraint construction (body)"
            lb = pyo.value(constr[c].lower)
            ub = pyo.value(constr[c].upper)
            assert ((lb is not None) or (ub is not None)), "Error in constraint construction (LHS/RHS)"
    
    ### Step 4: Finish out the Iteration by "solving subproblem", "performing gradient ###
    ### descent step", and calculating the learning rate (BMPS, 2018)
    online_cb.next_iteration(part_for_BMPS=2)
    
    ## Save some data ##
    c_t_dict_vecs[i] = copy.deepcopy(online_cb.c_t_BMPS)
    xbar_dict_vecs[i] = copy.deepcopy(online_cb.xbar_t_BMPS)
    #x_t_samples
    
pdb.set_trace()
    
print("Run time (seconds) for the script: ")             
time_1 = time.process_time() - time_0
print(time_1)      


#COME BACK TO HERE!!!!###################################################################

#NEED TO PRINT THE D
#Also need to make graphs!
#ALSO NEED TO BUILD IN A WAY TO JUST DEFAULT TO 1/sqrt(t)

#print("This is D:",online_cb.D)
#############################################################


##### Step 4: Graph the Loss???????????? #####


##### Step 4b: Graph BMPS 2018 Error Measures #####
true_u_vec = np.reshape(true_u_vec,(50,1))

BMPS_obj_error = np.zeros((num_samples,))
BMPS_sol_error = np.zeros((num_samples,))
BMPS_total_error = np.zeros((num_samples,))

for i in range(1,num_samples+1): #cannot directly compare graph to paper bc we use 1/sqrt(t) - might need to run for more iterations?
    BMPS_obj_error[i-1] = np.dot(np.transpose(c_t_dict_vecs[i]),(x_t_samples[i]-xbar_dict_vecs[i]))
    BMPS_sol_error[i-1] = np.dot(np.transpose(true_u_vec),(x_t_samples[i]-xbar_dict_vecs[i])) #since we made u_vec as like negative ok?
    BMPS_total_error[i-1] = BMPS_obj_error[i-1] - BMPS_sol_error[i-1] #screwed up because of my negative? BECAUSE OF THE ARG MIN
                    #THING AND MY NOT THINKING ABOUT THE OBJECTIVE FUNC THE WAY i SHOULD BE??
 
### Need to do the cumsum ###
BMPS_obj_error = np.divide(np.cumsum(BMPS_obj_error),np.arange(1,num_samples+1))
BMPS_sol_error = np.divide(np.cumsum(BMPS_sol_error),np.arange(1,num_samples+1))
BMPS_total_error = np.divide(np.cumsum(BMPS_total_error),np.arange(1,num_samples+1))
    
#################################    
plt.subplot(131)
error_graph = plt.plot(np.arange(1,num_samples+1),BMPS_obj_error)
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('Objective error: u_t^T (xbar_t-x_t)',fontsize=20)
plt.title('BMPS Objective Error',fontsize=20)

plt.subplot(132)
error_graph = plt.plot(np.arange(1,num_samples+1),BMPS_sol_error)
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('Solution error: u_true^T (xbar_t-x_t)',fontsize=20)
plt.title('BMPS Solution Error',fontsize=20)

plt.subplot(133)
error_graph = plt.plot(np.arange(1,num_samples+1),BMPS_total_error)
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('Total error: Obj Error - Sol Error',fontsize=20)
plt.title('BMPS Total Error',fontsize=20)


plt.show()

#bound_func = lambda x: 1/(math.sqrt(x)) #following the lead of BMPS 2018

#average_bound = plt.



        

#NEED TO ALSO MAKE A GRAPH WITH r!
#probably can just build that as another list attribute
#into the class 
